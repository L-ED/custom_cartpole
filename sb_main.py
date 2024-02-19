from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
import torch

from env import CustomCartpole

def main(test=True):

    env = CustomCartpole()
    agent_kwargs = {}
    trainer_kwargs = {}

    path = '/home/led/robotics/engines/Bullet_sym/custom_cartpole/results'
    new_logger = configure(path, ["stdout"])
    agent = PPO("MlpPolicy", env, device='cpu')
    agent.set_logger(new_logger)
    env.enable_suggestions = True
    agent.learn(total_timesteps=200000)
    if test:
        env = CustomCartpole("human")
        state, _ =  env.reset()
        while True:
            action, _state = agent.predict(state, deterministic=True)
            state, reward, term,trunc,_ = env.step(action)
            if term or trunc:
                state, _ =  env.reset()

if __name__=="__main__":
    main()