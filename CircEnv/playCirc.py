import gymnasium as gym
import numpy as np
import pygame
import circ_env

env = gym.make('circ_env/Circle-v0', render_mode="human")

# Reset the environment
obs = env.reset()
env.render()

# Run the environment with random actions
#for i in range(500):

start_pos = None
reset_time = 0

while True:

    #if any([event.type == pygame.QUIT for event in pygame.event.get()]): break
        
    #player controls
    keys = pygame.key.get_pressed() 
    if keys[pygame.K_LEFT]: x = -1
    elif keys[pygame.K_RIGHT]: x = 1
    else: x = 0
    if keys[pygame.K_UP]: y = -1
    elif keys[pygame.K_DOWN]: y = 1
    else: y = 0   

    action = np.array([x,y],dtype=np.float32)
    #action = env.action_space.sample()
    #print(action)
    obs, reward, done, _, _ = env.step(action)
    
    # Render the environment
    env.render()
    
    # Check if the episode is finished
    if done:
        obs = env.reset()
        start_pos = None
        reset_time = 0
        
# Close the environment
env.close()