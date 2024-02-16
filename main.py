import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt

from env import CustomCartpole
from agent import ActorCritic
from ppo import PPO

def main(test=False):

    env = CustomCartpole()
    agent_kwargs = {}
    trainer_kwargs = {}

    agent = ActorCritic(
        env.observation_space,
        env.action_space,
        **agent_kwargs
    )

    trainer = PPO(
        env,
        agent,
        **trainer_kwargs
    )
    trainer.learn()
    if test:
        env = CustomCartpole("human")
        state, _ =  env.reset()
        while True:
            action = agent(state)
            state,rew,term,trunc,_ = env.step(action)
            if term or trunc:
                state, _ =  env.reset()

if __name__=="__main__":
    main()