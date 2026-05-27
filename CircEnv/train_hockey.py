import argparse
import csv
import shutil
from pathlib import Path

import gymnasium as gym
from sb3_contrib import TQC
from stable_baselines3.common.evaluation import evaluate_policy
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


def build_model(env, resume=False):
    checkpoint = latest_checkpoint(MODELS_DIR) if resume else None
    if checkpoint is not None:
        print(f"Resuming from {checkpoint}")
        model = TQC.load(str(checkpoint), env=env)
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
    )
    return model, 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument("--iterations", type=int, default=0, help="0 pomeni, da tece do Ctrl+C.")
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--fresh", action="store_true", help="Zacne nov model namesto nadaljevanja checkpointa.")
    parser.add_argument("--resume", action="store_true", help="Ohranjeno zaradi kompatibilnosti; resume je privzet.")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(__file__, MODELS_DIR / Path(__file__).name)
    shutil.copy(ENV_FILE, MODELS_DIR / ENV_FILE.name)

    env = gym.make("circ_env/AirHockey-v0")
    model, start_steps = build_model(env, resume=not args.fresh)

    best_reward = -float("inf")
    rewards_csv = MODELS_DIR / "rewards.csv"
    best_rewards_csv = MODELS_DIR / "best_rewards.csv"

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
