import gymnasium as gym
import numpy as np
#from sb3_contrib import TQC
from stable_baselines3 import PPO
import os
import circ_env

env = gym.make('circ_env/Circle-v0')

# tensorboard logiranje
models_dir = "models/PPO_puck_dynamics"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
logdir = "logs_puck_dynamics"
if not os.path.exists(logdir):  
    os.makedirs(logdir)


# PPO01
model = PPO('MlpPolicy',
            env=env,
            tensorboard_log=logdir,
            verbose=1,
            n_steps=512,
            batch_size=256,
            gae_lambda=0.9,
            gamma=0.99,
            n_epochs=5,
            ent_coef=0.0,
            learning_rate=2.5e-4,
            clip_range=0.3,
            seed=2)
     
# Reset the environment
obs = env.reset()

# iteracija skozi učenje in shranjevanje modela
TIMESTEPS = 10000
iters = 0
while True:
    iters += 1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False,tb_log_name="PPO_puck_dynamics")

    model.save(f"{models_dir}/{TIMESTEPS*iters}")
