import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path

import gymnasium as gym
import numpy as np
from sb3_contrib import TQC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch
from torch import nn

import circ_env


SCRIPT_DIR = Path(__file__).resolve().parent
METHOD_DIR_NAME = "TQC_hockey_active_goal"
MODELS_DIR = SCRIPT_DIR / "models" / METHOD_DIR_NAME
LOG_DIR = SCRIPT_DIR / "logs_hockey_active_goal"
ENV_FILE = SCRIPT_DIR / "circ_env" / "envs" / "hockey_world.py"
PROFESSOR_MODEL_PATHS = [
    SCRIPT_DIR / "models" / "TQC02" / "280000",
    SCRIPT_DIR.parent / "models" / "TQC02" / "280000",
]
TERMINATION_REASONS = [
    "agent_goal",
    "opponent_goal",
    "agent_border",
    "agent_table_border",
    "agent_hockey_border",
    "agent_center_line",
    "timeout",
    "time_limit",
    "unknown",
]


def latest_checkpoint(models_dir):
    checkpoints = [
        path for path in models_dir.glob("*.zip")
        if path.stem.isdigit()
    ]
    if checkpoints:
        return max(checkpoints, key=lambda path: int(path.stem))

    best = models_dir / "best.zip"
    if best.exists():
        return best

    return None


def checkpoint_step(checkpoint, models_dir):
    if checkpoint.stem.isdigit():
        return int(checkpoint.stem)

    best_rewards_csv = models_dir / "best_rewards.csv"
    if best_rewards_csv.exists():
        with best_rewards_csv.open(newline="") as csvfile:
            rows = list(csv.reader(csvfile))
        for row in reversed(rows[1:]):
            try:
                return int(float(row[0]))
            except (ValueError, IndexError):
                continue

    return 0


def append_csv(path, row, header):
    new_file = not path.exists()
    with path.open("a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if new_file:
            writer.writerow(header)
        writer.writerow(row)


def evaluate_termination_reasons(model, episodes):
    reasons = Counter()
    eval_env = gym.make("circ_env/AirHockey-v0")
    try:
        for _ in range(episodes):
            obs, _ = eval_env.reset()
            done = False
            last_info = {}
            while not done:
                action, _state = model.predict(obs, deterministic=True)
                obs, _reward, terminated, truncated, last_info = eval_env.step(
                    np.asarray(action).squeeze()
                )
                done = terminated or truncated

            reason = last_info.get("termination_reason")
            if reason is None and truncated:
                reason = "time_limit"
            reasons[reason or "unknown"] += 1
    finally:
        eval_env.close()

    return reasons


def resolve_device(requested_device):
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"

    if requested_device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        return "cpu"

    if requested_device == "cuda":
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU.")

    return requested_device


def make_training_env(num_envs):
    if num_envs <= 1:
        return gym.make("circ_env/AirHockey-v0")

    return make_vec_env(
        "circ_env/AirHockey-v0",
        n_envs=num_envs,
        vec_env_cls=SubprocVecEnv,
    )


def num_envs(env):
    return getattr(env, "num_envs", 1)


def build_model(env, resume=False, device="auto"):
    checkpoint = latest_checkpoint(MODELS_DIR) if resume else None
    if checkpoint is not None:
        print(f"Resuming from {checkpoint}")
        model = TQC.load(
            str(checkpoint),
            env=env,
            device=device,
            custom_objects={"n_envs": num_envs(env)},
        )
        model.tensorboard_log = str(LOG_DIR)
        return model, checkpoint_step(checkpoint, MODELS_DIR)

    policy_kwargs = dict(
        n_critics=2,
        n_quantiles=25,
        activation_fn=nn.Tanh,
        net_arch=[256, 256],
    )
    model = TQC(
        "MlpPolicy",
        env=env,
        tensorboard_log=str(LOG_DIR),
        verbose=1,
        policy_kwargs=policy_kwargs,
        buffer_size=300_000,
        batch_size=256,
        learning_starts=2_000,
        train_freq=1,
        gradient_steps=1,
        gamma=0.995,
        learning_rate=3e-4,
        tau=0.02,
        ent_coef="auto",
        device=device,
    )
    return model, 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument("--iterations", type=int, default=0, help="0 pomeni, da tece do Ctrl+C.")
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--envs", type=int, default=1, help="Stevilo paralelnih okolij za zbiranje izkusenj.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--fresh", action="store_true", help="Zacne nov model namesto nadaljevanja checkpointa.")
    parser.add_argument("--resume", action="store_true", help="Ohranjeno zaradi kompatibilnosti; resume je privzet.")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(__file__, MODELS_DIR / Path(__file__).name)
    shutil.copy(ENV_FILE, MODELS_DIR / ENV_FILE.name)

    device = resolve_device(args.device)
    env = make_training_env(args.envs)
    print(f"Using {args.envs} training environment(s).")
    model, start_steps = build_model(env, resume=not args.fresh, device=device)

    best_reward = -float("inf")
    rewards_csv = MODELS_DIR / "rewards.csv"
    best_rewards_csv = MODELS_DIR / "best_rewards.csv"
    termination_reasons_csv = MODELS_DIR / "termination_reasons.csv"

    iters = 0
    while args.iterations == 0 or iters < args.iterations:
        iters += 1
        model.learn(
            total_timesteps=args.timesteps,
            reset_num_timesteps=False,
            tb_log_name=METHOD_DIR_NAME,
        )

        total_steps = start_steps + args.timesteps * iters
        mean_reward, std_reward = evaluate_policy(
            model,
            model.get_env(),
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
        )

        append_csv(
            rewards_csv,
            [total_steps, mean_reward, std_reward],
            ["step", "mean reward", "std reward"],
        )

        termination_reasons = evaluate_termination_reasons(model, args.eval_episodes)
        termination_row = [total_steps] + [
            termination_reasons.get(reason, 0)
            for reason in TERMINATION_REASONS
        ]
        append_csv(
            termination_reasons_csv,
            termination_row,
            ["step"] + TERMINATION_REASONS,
        )
        reason_summary = ", ".join(
            f"{reason}={termination_reasons.get(reason, 0)}"
            for reason in TERMINATION_REASONS
        )
        print(f"Termination reasons: {reason_summary}")

        checkpoint_path = MODELS_DIR / str(total_steps)
        model.save(str(checkpoint_path))
        print(f"Saved {checkpoint_path}.zip, mean reward {mean_reward:.3f}")

        if mean_reward > best_reward:
            best_reward = mean_reward
            best_path = MODELS_DIR / "best"
            model.save(str(best_path))
            for professor_model_path in PROFESSOR_MODEL_PATHS:
                professor_model_path.parent.mkdir(parents=True, exist_ok=True)
                model.save(str(professor_model_path))
            append_csv(
                best_rewards_csv,
                [total_steps, mean_reward, std_reward],
                ["step", "mean reward", "std reward"],
            )
            print(f"New best model: {best_path}.zip")
            print("Updated professor paths:")
            for professor_model_path in PROFESSOR_MODEL_PATHS:
                print(f"  {professor_model_path}.zip")

    env.close()


if __name__ == "__main__":
    main()
