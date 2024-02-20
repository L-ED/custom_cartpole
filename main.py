import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
import torch

from env import CustomCartpole
from agent import ActorCritic
from ppo import PPO

def main(test=True):

    env = CustomCartpole()
    agent_kwargs = {}
    trainer_kwargs = {}

    agent = ActorCritic(
        env.observation_space,
        env.action_space,
        **agent_kwargs
    )

    env.enable_suggestions=True
    trainer = PPO(
        env,
        agent,
        **trainer_kwargs
    )
    trainer.learn()
    trainer.load_best_weights()
    if test:
        env = CustomCartpole("human")
        state, _ =  env.reset()
        while True:
            state = env.normalize_state(state)
            state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            # action, val, log_prob = agent.step(state)
            action = agent.act(state)

            state,rew,term,trunc,_ = env.step(action.item())
            if term or trunc:
                state, _ =  env.reset()

if __name__=="__main__":
    main()