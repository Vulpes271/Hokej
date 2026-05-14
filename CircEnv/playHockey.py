import gymnasium as gym
import numpy as np
import pygame
import circ_env

env = gym.make('circ_env/AirHockey-v0', render_mode="human")

# Reset the environment
obs = env.reset()
env.render()

# Run the environment with random actions
#for i in range(500):
while True:

    if any([event.type == pygame.QUIT for event in pygame.event.get()]): break
        
    agent_vel = env.action_space.high[0]
    #player controls
    x = 0
    y = 0
    keys = pygame.key.get_pressed() 
    if keys[pygame.K_LEFT]: x = -agent_vel
    elif keys[pygame.K_RIGHT]: x = agent_vel
    else: x = 0
    if keys[pygame.K_UP]: y = -agent_vel
    elif keys[pygame.K_DOWN]: y = agent_vel
    else: y = 0  

      
    action = np.array([x,y],dtype=np.float32)

    #action = env.action_space.sample()
    #action = np.array([-1.0, -1.0])
    #print(action)
    obs, reward, done, _, _ = env.step(action)
    
    # Render the environment
    env.render()
    
    # Check if the episode is finished
    if done:
        obs = env.reset()
        
# Close the environment
env.close()