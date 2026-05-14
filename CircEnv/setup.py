from setuptools import setup

setup(
    name="circ_env",
    version="0.0.1",
    author="Janez Podobnik",
    install_requires=["gymnasium", "pygame", "box2d", "stable-baselines3", "sb3_contrib", "tensorboard"]
)