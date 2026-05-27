from setuptools import find_packages, setup

setup(
    name="circ_env",
    version="0.0.1",
    author="Janez Podobnik",
    packages=find_packages(include=["circ_env", "circ_env.*"]),
    install_requires=[
        "gymnasium",
        "pygame",
        "box2d",
        "stable-baselines3",
        "sb3-contrib",
        "tensorboard",
    ],
)
