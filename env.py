from gymnasium.envs.classic_control.cartpole import CartPoleEnv
import numpy as np
from gymnasium import logger, spaces



class CustomCartpole(CartPoleEnv):

    def __init__(self, render_mode: str | None = None):
        super().__init__(render_mode)

        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,#self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(-high, high, dtype=np.float32)


    def reset(self, *, seed: int | None = None, options: dict | None = None):
        state, info = super().reset(seed=seed, options=options)
        state[2] = np.pi
        self.state = state
        return state , info