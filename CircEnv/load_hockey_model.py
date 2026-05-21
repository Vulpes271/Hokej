from pathlib import Path

import gymnasium as gym
import pygame
from sb3_contrib import TQC

import circ_env


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIRS = [
    SCRIPT_DIR / "models" / "TQC_hockey",
    SCRIPT_DIR / "models" / "TQC01",
    SCRIPT_DIR.parent / "models" / "TQC_hockey",
    SCRIPT_DIR.parent / "models" / "TQC01",
]


def latest_model_path():
    candidates = []
    for models_dir in MODEL_DIRS:
        if not models_dir.exists():
            continue
        candidates.extend(path for path in models_dir.glob("*.zip") if path.stem.isdigit())
        best = models_dir / "best.zip"
        if best.exists():
            candidates.append(best)

    if not candidates:
        searched = "\n".join(str(path) for path in MODEL_DIRS)
        raise FileNotFoundError(f"Ni TQC hokej modela. Pregledane mape:\n{searched}")

    numeric = [path for path in candidates if path.stem.isdigit()]
    if numeric:
        return max(numeric, key=lambda path: int(path.stem))
    return candidates[0]


def main():
    env = gym.make("circ_env/AirHockey-v0", render_mode="human")
    model_path = latest_model_path()
    print(f"Loading model: {model_path}")
    model = TQC.load(str(model_path), env=env)

    obs, _ = env.reset()

    while True:
        if any(event.type == pygame.QUIT for event in pygame.event.get()):
            break

        action, _state = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action.squeeze())
        env.render()

        if terminated or truncated:
            obs, _ = env.reset()

    env.close()


if __name__ == "__main__":
    main()
