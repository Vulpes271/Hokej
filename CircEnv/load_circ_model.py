import gymnasium as gym
import numpy as np
#from sb3_contrib import TQC
from stable_baselines3 import PPO
import stable_baselines3.common.base_class as sb3_base_class
import stable_baselines3.common.save_util as sb3_save_util
import os
from pathlib import Path
import tempfile
import zipfile
import torch as th
import circ_env


def load_from_zip_file_disk(load_path, load_data=True, custom_objects=None, device="auto", verbose=0, print_system_info=False):
    device = sb3_save_util.get_device(device=device)
    data = None
    pytorch_variables = None
    params = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(load_path) as archive:
            namelist = archive.namelist()

            if print_system_info and "system_info.txt" in namelist:
                print("== SAVED MODEL SYSTEM INFO ==")
                print(archive.read("system_info.txt").decode())

            if "data" in namelist and load_data:
                json_data = archive.read("data").decode()
                data = sb3_save_util.json_to_data(json_data, custom_objects=custom_objects)

            for file_path in [name for name in namelist if os.path.splitext(name)[1] == ".pth"]:
                archive.extract(file_path, tmpdir)
                th_object = th.load(os.path.join(tmpdir, file_path), map_location=device, weights_only=True)
                if file_path == "pytorch_variables.pth" or file_path == "tensors.pth":
                    pytorch_variables = th_object
                else:
                    params[os.path.splitext(file_path)[0]] = th_object

    return data, params, pytorch_variables


sb3_save_util.load_from_zip_file = load_from_zip_file_disk
sb3_base_class.load_from_zip_file = load_from_zip_file_disk

env = gym.make('circ_env/Circle-v0', render_mode="human")

models_dir = Path("models/PPO_puck_dynamics")
model_path = max((path for path in models_dir.glob("*.zip") if path.stem.isdigit()), key=lambda path: int(path.stem))
print(f"Loading model: {model_path}")
model = PPO.load(str(model_path), env=env)

# Reset the environment
vec_env = model.get_env()
obs = vec_env.reset()

EPISODES = 1000

for episode in range(EPISODES):

    obs = vec_env.reset()
    done = False

    while not done:

        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action.squeeze())

        env.render()
        
# Close the environment
env.close()
vec_env.close()
