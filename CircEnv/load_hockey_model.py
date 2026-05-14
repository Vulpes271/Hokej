import gym
import numpy as np
from sb3_contrib import TQC
#from stable_baselines3 import PPO
import os
import circ_env

env = gym.make('circ_env/AirHockey-v0', render_mode="human")

model = TQC.load("./models/TQC01/7420000", env=env)

# Reset the environment
vec_env = model.get_env()
obs = vec_env.reset()

EPISODES = 1000

#for episode in range(EPISODES):
while True:

    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)
    env.render()

    # Check if the episode is finished
    if done:
        obs = env.reset()
        
# Close the environment
env.close()
vec_env.close()       