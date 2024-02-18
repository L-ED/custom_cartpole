import torch
from torch.optim import Adam
import numpy as np
import scipy

import torch.nn.functional as func
from torch.utils.data import Dataset, DataLoader

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

        with torch.no_grad():
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
        # self.normalize_advantage()
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
        self.adv = torch.from_numpy(self.adv).float()
        self.rtg = torch.from_numpy(self.rtg).float()
        
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

    def __init__(self, env, agent, seed=0, epochs=200000, batch_size=64, iters_per_epoch=4000, optim_iters=10, clip_eps=0.2, max_grad_norm=0.5, vf_koef=0.5) -> None:

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.vf_koef = vf_koef
        self.batch_size = batch_size
        self.agent = agent
        self.env = env
        self.epochs = epochs
        self.epoch_iters = iters_per_epoch
        self.clip_eps = clip_eps
        self.optimization_iters = optim_iters

        self.max_grad_norm = max_grad_norm

        self.buffer = Buffer(
            obs_space= env.observation_space.shape[0]
        )


        # self.policy_optim = Adam(self.agent.parameters(), lr= 1e-2)

        if self.agent.shared:
            self.policy_optim = Adam(self.agent.parameters(), lr= 5e-4)
            self.value_optim=None
        else:
            self.policy_optim = Adam(self.agent.policy.parameters(), lr= 1e-3)
            self.value_optim = Adam(self.agent.value.parameters(), lr=1e-4)
    

    def learn(self):
        state, info = self.env.reset()
        state = self.env.normalize_state(state)
        for epoch in range(self.epochs):
            traj_num = 0
            reward_collector = 0
            for iteration in range(self.epoch_iters):
                state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
                action, val, log_prob = self.agent.step(state)
                next_state,rew,term,trunc,_ = self.env.step(action.item())
                self.buffer.collect(state, action, val, rew, log_prob)
                state = next_state
                state = self.env.normalize_state(state)

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
            
            msg = f"Epoch {epoch}: traj avg_rew {reward_collector/traj_num}, avg_traj_len {self.epoch_iters/traj_num}"
            print(msg)

            # self.update()
            # self.update_iteratively()
            self.update_manual()


    def update(self):

        data=self.buffer.get()

        loss = self.loss(data)
        loss.backward()

        if self.value_optim is not None:
            self.value_optim.step()
            self.value_optim.zero_grad()

        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
        self.policy_optim.step()
        self.policy_optim.zero_grad()

        self.buffer.reset()

        
    def loss(self, data):
        
        states, act, rtg, adv, log_prob_old = data 

        adv = (adv - adv.mean())/adv.std()

        pi, log_prob = self.agent.get_probs(states, act)
        ratio = torch.exp(log_prob - log_prob_old)
        ratio_cliped = torch.clamp(
            ratio, 1-self.clip_eps, 1+self.clip_eps)
        p_loss = -torch.min(ratio*adv, ratio_cliped*adv).mean()

        # if self.agent.shared:
        # p_loss -= pi.entropy().mean()

        v_loss = func.mse_loss(self.agent.value(states).flatten(), rtg)

        return p_loss, v_loss
    

    def create_dataloader(self, data):

        return DataLoader(
            BufferDataset(data),
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=2
        )
    
    def update_manual(self):

        p_loss_col, v_loss_col = np.zeros(0), np.zeros(0)

        states, act, rtg, adv, log_prob_old=self.buffer.get()
        b_inds = np.arange(len(rtg))
        np.random.shuffle(b_inds)
        for start in range(0, len(rtg), self.batch_size):
            sample_idx = b_inds[start:start+64]
            data = [
                states[sample_idx], 
                act[sample_idx], 
                rtg[sample_idx], 
                adv[sample_idx], 
                log_prob_old[sample_idx]
            ] 
            p_loss, v_loss = self.loss(data)
            loss = p_loss + v_loss*self.vf_koef
            loss.backward()

            p_loss_col = np.append(p_loss_col, p_loss.item()) 
            v_loss_col = np.append(v_loss_col, v_loss.item())

            if self.value_optim is not None:
                self.value_optim.step()
                self.value_optim.zero_grad()

            torch.nn.utils.clip_grad_norm_(self.agent.policy.parameters(), self.max_grad_norm)
            self.policy_optim.step()
            self.policy_optim.zero_grad()

        print("v_loss", v_loss_col.mean(), "p_loss", p_loss_col.mean())

        self.buffer.reset()









#     def update_iteratively(self):

#         dataloader = self.create_dataloader(
#             self.buffer.get())

#         for data in dataloader:

#             loss = self.loss(data)
#             loss.backward()

#             # if self.value_optim is not None:
#                 # self.value_optim.step()
#                 # self.value_optim.zero_grad()

#             torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
#             self.policy_optim.step()
#             self.policy_optim.zero_grad()

#         self.buffer.reset()


# class BufferDataset(Dataset):

#     def __init__(self, data):
#         super().__init__()

#         self.states, self.act, self.rtg, self.adv, self.log_prob = data

#     def __getitem__(self, index):
#         return self.states[index], self.act[index], self.rtg[index], self.adv[index], self.log_prob[index]

#     def __len__(self):
#         return len(self.rtg)