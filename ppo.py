import torch
from torch.optim import Adam
import numpy as np
import scipy

class Buffer:
    # folder for collecting stat-acttion pairs
    def __init__(self, gamma, lambd) -> None:
        self.gamma = gamma
        self.lambd = lambd
        self.sts = np.zeros(0)
        self.act = np.zeros(0)
        self.rew = np.zeros(0)
        self.val = np.zeros(0)
        self.log_probs = np.zeros(0)

        self.traj_lenght= 0

        self.rtg = np.zeros(0) #np.array([])
        self.adv = np.zeros(0) #np.array([])


    def collect(self, state, action, value, reward, log_prob):

        self.sts = np.append(self.sts, state)
        self.act = np.append(self.act, action)
        self.rew = np.append(self.rew, reward)
        self.val = np.append(self.val, value)
        self.log_probs = np.append(self.log_probs, log_prob)
    
        self.traj_lenght +=1

    
    def end_traj(self, last_val):

        rew = self.rew[-self.traj_lenght:]
        val = self.val[-self.traj_lenght:]

        rew = np.append(rew, last_val)
        val = np.append(val, last_val)

        delta = rew[:-1] + self.gamma*val[1:] - val[:-1]

        adv = self.discounted_sum(delta, self.gamma*self.lambd)
        rtg = self.discounted_sum(delta, self.gamma)

        self.adv = np.append(self.adv, adv)
        self.rtg = np.append(self.rtg, rtg)


    def discounted_sum(sequence, coef):
        return scipy.signal.lfilter(
            [1], [1, float(-coef)], sequence[::-1], axis=0)[::-1]


    def get(self):
        self.normalize_advantage()
        data = [
            self.sts, 
            self.act, 
            self.rtg,
            self.adv, 
            self.log_prob
        ]
        
        return [torch.from_numpy(v) for v in data]


    def normalize_advantage(self):
        mean, std = self.adv.mean(), self.adv.std()
        self.adv = (self.adv - mean)/std



class PPO:

    def __init__(self, env, agent, epochs, iters_per_epoch, optim_iters, clip_eps=0.2) -> None:

        self.agent = agent
        self.env = env
        self.epochs = epochs
        self.epoch_iters = iters_per_epoch
        self.clip_eps = clip_eps
        self.optimization_iters = optim_iters
        self.buffer = Buffer()

        if self.agent.shared:
            self.policy_optim = Adam(self.agent)
        else:
            self.policy_optim = Adam(self.agent.policy)
            self.value_optim = Adam(self.agent.value)
    

    def learn(self):
        state, info = self.env.reset()
        for epoch in self.epochs:
            traj_num = 0
            reward_collector = 0
            for iteration in self.epoch_iters:
                action, val, log_prob = self.agent(state)
                next_state,rew,term,trunc,_ = self.env.step()
                self.buffer.collect(state, action, val, rew, log_prob)
                state = next_state

                last_iter = iteration == self.epoch_iters-1
                reward_collector += rew
                if term or trunc or last_iter:
                    if term:
                        val = 0
                    else:
                        _, val, _= self.agent(state)
                
                    self.buffer.end_traj(val)
                    state, info = self.env.reset()
                    traj_num +=1
            
            msg = f"Epoch {epoch}: traj avg_rew {reward_collector/traj_num}"
            print(msg)

            self.update()


    def update(self):

        data=self.buffer.get()

        for i in range(self.optimization_iters):
            p_loss, v_loss = self.loss(data)

            if self.agent.shared:
                p_loss += v_loss
            else:
                v_loss.backward()
                self.value_optim.step()
                self.val_optim.zero_grad()

            p_loss.backward()
            self.policy_optim.step()
            self.policy_optim.zero_grad()

        
    def loss(self, data):
        
        states, act, rtg, adv, log_prob_old = data 

        pi, log_prob = self.agent.policy(states, act)
        ratio = torch.exp(log_prob - log_prob_old)
        ratio_cliped = torch.clamp(
            ratio, 1-self.clip_eps, 1+self.clip_eps)
        p_loss = -torch.min(ratio*adv, ratio_cliped*adv).mean()

        if self.agent.shared:
            p_loss += pi.entropy()

        v_loss = ((self.agent.value(states) - rtg)**2).mean()

        return p_loss, v_loss
    
