import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
import torch

from env import CustomCartpole
from agent import ActorCritic
from ppo import PPO

def main():

    env = CustomCartpole("human")

    agent = ActorCritic(
        env.observation_space,
        env.action_space,
    )

    trainer = PPO(
        env,agent
    )
    trainer.load_best_weights()
    state, _ =  env.reset()
    while True:
        state = env.normalize_state(state)
        state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        action = agent.act(state)

        state,rew,term,trunc,_ = env.step(action.item())
        if term or trunc:
            state, _ =  env.reset()


if __name__=="__main__":
    main()