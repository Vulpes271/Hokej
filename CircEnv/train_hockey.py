import gymnasium as gym
import numpy as np
from sb3_contrib import TQC
#from stable_baselines3 import PPO
import os
import circ_env


env = gym.make('circ_env/AirHockey-v0')

# tensorboard logiranje
models_dir = "models/TQC01"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
logdir = "logs"
if not os.path.exists(logdir):  
    os.makedirs(logdir)

# deklaracija modela
policy_kwargs = dict(n_critics=2, n_quantiles=25)
model = TQC('MlpPolicy', env=env, tensorboard_log=logdir, verbose=1,policy_kwargs=policy_kwargs)

# Reset the environment
obs = env.reset()

# iteracija skozi učenje in shranjevanje modela
TIMESTEPS = 10000
iters = 0
while True:
    iters += 1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False,tb_log_name="TQC_01")

    model.save(f"{models_dir}/{TIMESTEPS*iters}")
