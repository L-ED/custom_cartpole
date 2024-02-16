import torch
from torch.optim import Adam
import numpy as np
import scipy

class Buffer:
    # folder for collecting stat-acttion pairs
    def __init__(self, obs_space, gamma=0.99, lambd=0.95) -> None:
        self.gamma = gamma
        self.lambd = lambd
        self.obs_space = obs_space
        self.reset()
        # # self.sts = np.zeros((0, obs_space), dtype=np.float32)
        # # self.act = np.zeros(0, dtype=np.float32)
        # self.rew = np.zeros(0, dtype=np.float32)
        # # self.val = np.zeros(0, dtype=np.float32)
        # self.sts = torch.zeros((0, obs_space))
        # self.act = torch.zeros(0)
        # # self.rew = torch.zeros(0)
        # self.val = torch.zeros(0)
        # self.log_probs = torch.zeros(0)

        # self.traj_lenght= 0

        # self.rtg = np.zeros(0, dtype=np.float32) #np.array([])
        # self.adv = np.zeros(0, dtype=np.float32) #np.array([])


    def collect(self, state, action, value, reward, log_prob):

        # self.sts = np.vstack((
        #     self.sts,
        #     state 
        # ))

        # self.sts = np.append(self.sts, state)
        # self.act = np.append(self.act, action)
        self.rew = np.append(self.rew, reward)
        # self.val = np.append(self.val, value)

        self.sts = torch.cat((self.sts, state))
        self.act = torch.cat((self.act, action))
        # self.rew = torch.cat((self.rew, reward))
        self.val = torch.cat((self.val, value))
        self.log_probs = torch.cat((self.log_probs, log_prob.unsqueeze(0)))
    
        self.traj_lenght +=1

    
    def end_traj(self, last_val):

        rew = self.rew[-self.traj_lenght:]#.numpy()
        val = self.val[-self.traj_lenght:].detach().numpy()

        rew = np.append(rew, last_val)
        val = np.append(val, last_val)

        delta = rew[:-1] + self.gamma*val[1:] - val[:-1]

        adv = self.discounted_sum(delta, self.gamma*self.lambd)
        rtg = self.discounted_sum(delta, self.gamma)

        self.adv = np.append(self.adv, adv)
        self.rtg = np.append(self.rtg, rtg)

        self.traj_lenght =0


    def discounted_sum(self, sequence, coef):
        return scipy.signal.lfilter(
            [1], [1, float(-coef)], sequence[::-1], axis=0)[::-1]


    def get(self):
        self.normalize_advantage()
        # data = [
        #     self.sts, 
        #     self.act, 
        #     self.rtg,
        #     self.adv, 
        #     # self.log_prob
        # ]
        # data = [torch.from_numpy(v) for v in data]
        # data.append(self.log_probs)
        # return data
        self.adv = torch.from_numpy(self.adv)
        self.rtg = torch.from_numpy(self.rtg)
        
        return self.sts, self.act, self.rtg, self.adv, self.log_probs
    

    def reset(self):

        # self.sts = np.zeros((0, obs_space), dtype=np.float32)
        # self.act = np.zeros(0, dtype=np.float32)
        self.rew = np.zeros(0, dtype=np.float32)
        # self.val = np.zeros(0, dtype=np.float32)
        self.sts = torch.zeros((0, self.obs_space))
        self.act = torch.zeros(0)
        # self.rew = torch.zeros(0)
        self.val = torch.zeros(0)
        self.log_probs = torch.zeros(0)

        self.traj_lenght= 0

        self.rtg = np.zeros(0, dtype=np.float32) #np.array([])
        self.adv = np.zeros(0, dtype=np.float32) #np.array([])


    def normalize_advantage(self):
        mean, std = self.adv.mean(), self.adv.std()
        self.adv = (self.adv - mean)/std



class PPO:

    def __init__(self, env, agent, seed=0, epochs=5000, iters_per_epoch=4000, optim_iters=80, clip_eps=0.2) -> None:

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.agent = agent
        self.env = env
        self.epochs = epochs
        self.epoch_iters = iters_per_epoch
        self.clip_eps = clip_eps
        self.optimization_iters = optim_iters
        self.buffer = Buffer(
            obs_space= env.observation_space.shape[0]
        )

        if self.agent.shared:
            self.policy_optim = Adam(self.agent.parameters(), lr= 5e-4)
            self.value_optim=None
        else:
            self.policy_optim = Adam(self.agent.policy.parameters(), lr= 3e-4)
            self.value_optim = Adam(self.agent.value.parameters(), lr=1e-3)
    

    def learn(self):
        state, info = self.env.reset()
        for epoch in range(self.epochs):
            traj_num = 0
            reward_collector = 0
            for iteration in range(self.epoch_iters):
                state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)

                action, val, log_prob = self.agent.step(state)
                next_state,rew,term,trunc,_ = self.env.step(action.item())
                self.buffer.collect(state, action, val, rew, log_prob)
                state = next_state

                last_iter = iteration == self.epoch_iters-1
                reward_collector += rew
                if term or trunc or last_iter:
                    if term:
                        val = 0
                    else:
                        _, val, _= self.agent.step(
                            torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))
                        val = val.item()

                    self.buffer.end_traj(val)
                    state, info = self.env.reset()
                    traj_num +=1
            
            msg = f"Epoch {epoch}: traj avg_rew {reward_collector/traj_num}"
            print(msg)

            self.update()


    def update(self):

        data=self.buffer.get()

        loss = self.loss(data)
        loss.backward()

        if self.value_optim is not None:
            self.value_optim.step()
            self.value_optim.zero_grad()

        self.policy_optim.step()
        self.policy_optim.zero_grad()

        self.buffer.reset()

        
    def loss(self, data):
        
        states, act, rtg, adv, log_prob_old = data 

        pi, log_prob = self.agent.get_probs(states, act)
        ratio = torch.exp(log_prob - log_prob_old)
        ratio_cliped = torch.clamp(
            ratio, 1-self.clip_eps, 1+self.clip_eps)
        p_loss = -torch.min(ratio*adv, ratio_cliped*adv).mean()

        if self.agent.shared:
            p_loss -= pi.entropy()

        v_loss = ((self.agent.value(states) - rtg)**2).mean()

        return p_loss+v_loss
    
