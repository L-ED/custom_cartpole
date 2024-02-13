import gymnasium as gym
# from .policy import PPO 
import numpy as np
from matplotlib import pyplot as plt
# from pyvirtualdisplay import Display
from env import CustomCartpole

def main(test=False):
    # display = Display(visible=False, size=(1400, 900))
    # _ = display.start()

    # env = gym.make('CartPole-v1',  render_mode="human")
    env = CustomCartpole("human")
    print(env.reset())
    env.render()
    act = False
    while True:
        state,_,_,_,_ = env.step(0)
        print(state)
        # print("Norm", env.normalize_state(np.copy(state)))
        env.render()
        env.step(1)
        env.render()
        # plt.imshow(env.render())
        # img = plt.imshow(env.render())
        # img
        # imgenv.render()
    # policy = PPO(env)
    # success = policy.train()
    # if success:
    #     policy.validate()
    # if test:
    #     policy.test()


if __name__=="__main__":
    main()