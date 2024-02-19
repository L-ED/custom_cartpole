from gymnasium.envs.classic_control.cartpole import CartPoleEnv
import numpy as np
from gymnasium import logger, spaces
import math


class CustomCartpole(CartPoleEnv):
    # positive rotation 
    def __init__(self, render_mode = None):
        super().__init__(render_mode)

        self.x_threshold = 5
        self.x_dot_threshold = 15
        self.theta_dot_threshold = 15

        self.max_episode_steps = 500
        self.enable_suggestions = False

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


    def reset(self, *, seed = None, options = None):
        state, info = super().reset(seed=seed, options=options)
        state[2] = +np.pi
        if self.enable_suggestions:
            state[3] = np.random.uniform(-self.theta_dot_threshold/2, self.theta_dot_threshold/2)
        self.state = state
        self.steps = 0

        return state , info


    def step(self, action):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        ang_val = abs(theta)
        if ang_val > np.pi:
            theta = -np.sign(theta)*(np.pi - ang_val%np.pi)

        self.state = (x, x_dot, theta, theta_dot)

        # terminated = False
        terminated = bool(
            abs(x) > self.x_threshold
            or abs(x_dot) > self.x_dot_threshold
            or abs(theta_dot) > self.theta_dot_threshold
        )

        self.steps +=1
        truncated = self.steps >= self.max_episode_steps

        if not terminated:
            reward = self.calc_reward()#1.0
        else:
            reward = 0.0

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), reward, terminated, truncated, {}


    def calc_reward(self):
        ang_rew = np.cos(self.state[2])
        norm_state = self.normalize_state(np.copy(self.state))
        # moment_rew = np.log10(0.1+norm_state[2]**2)*np.log10(0.1+norm_state[3]**2)
        moment_rew = (1-norm_state[2]**2)*(1-norm_state[3]**2)#+((norm_state[3]*norm_state[2])**2)/2 
        return moment_rew + ang_rew  
    

    def normalize_state(self, state):
        state[0] /= self.x_threshold
        state[1] /= self.x_dot_threshold
        state[2] /= np.pi
        state[3] /= self.theta_dot_threshold
        return state

