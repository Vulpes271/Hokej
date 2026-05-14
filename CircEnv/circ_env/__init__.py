from gymnasium.envs.registration import register

register(
     id="circ_env/Circle-v0",
     entry_point="circ_env.envs:CircleEnvironment",
     max_episode_steps=500,
)

register(
     id="circ_env/AirHockey-v0",
     entry_point="circ_env.envs:HockeyEnv",
     max_episode_steps=500,
)