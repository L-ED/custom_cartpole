import torch
from torch.optim import Adam
import numpy as np
import scipy
import os

import torch.nn.functional as func
from torch.utils.data import Dataset, DataLoader

class Buffer:
    # folder for collecting stat-acttion pairs
    def __init__(self, obs_space, gamma=0.99, lambd=0.97) -> None:
        self.gamma = gamma
        self.lambd = lambd
        self.obs_space = obs_space
        self.reset()


    def collect(self, state, action, value, reward, log_prob):

        self.rew = np.append(self.rew, reward)
        self.sts = torch.cat((self.sts, state))
        self.act = torch.cat((self.act, action))
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
            rtg = self.discounted_sum(rew, self.gamma)[:-1]

            self.adv = np.append(self.adv, adv)
            self.rtg = np.append(self.rtg, rtg)
            # self.rtg = np.append(self.rtg, adv+val[:-1])

            self.traj_lenght =0


    def discounted_sum(self, sequence, coef):
        return scipy.signal.lfilter(
            [1], [1, float(-coef)], sequence[::-1], axis=0)[::-1]


    def get(self):

        self.adv = torch.from_numpy(self.adv).float()
        self.rtg = torch.from_numpy(self.rtg).float()
        
        return self.sts, self.act, self.rtg, self.adv, self.log_probs
    

    def reset(self):

        self.rew = np.zeros(0, dtype=np.float32)
        self.sts = torch.zeros((0, self.obs_space))
        self.act = torch.zeros(0)
        self.val = torch.zeros(0)
        self.log_probs = torch.zeros(0)

        self.traj_lenght= 0

        self.rtg = np.zeros(0, dtype=np.float32)
        self.adv = np.zeros(0, dtype=np.float32)


    def normalize_advantage(self):
        mean, std = self.adv.mean(), self.adv.std()
        self.adv = (self.adv - mean)/std



class PPO:

    def __init__(
            self, 
            env, 
            agent, 
            seed=0, 
            epochs=200, 
            batch_size=64, 
            iters_per_epoch=8000, 
            optim_iters=10, 
            clip_eps=0.2, 
            max_grad_norm=0.5, 
            vf_koef=0.5, 
            entropy_kf = 0.0,
            save_folder=None, 
            save_best=True,
            shared=True
        ) -> None:

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.save_best=save_best
        self.save_folder = save_folder
        if self.save_folder is None:
            self.save_folder = '/home/led/Simulators/Bullet/cartpole/weights'

        self.entropy_kf = entropy_kf
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

        if shared:
            self.policy_optim = Adam(self.agent.parameters(), lr= 1e-3)
            self.value_optim=None
        else:
            self.policy_optim = Adam(self.agent.policy.parameters(), lr= 3e-4)
            self.value_optim = Adam(self.agent.value.parameters(), lr=1e-3)
    

    def learn(self):
        state, info = self.env.reset()
        state = self.env.normalize_state(state)

        best_rew = -float('inf')

        for epoch in range(self.epochs):
            # if epoch == self.epochs/2:
            #     self.env.enable_suggestions=False
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
            
            cur_avg_rew = reward_collector/traj_num
            msg = f"Epoch {epoch}: traj avg_rew {cur_avg_rew}, avg_traj_len {self.epoch_iters/traj_num}"
            print(msg)

            if self.save_best:
                if cur_avg_rew>best_rew:
                    savepath=os.path.join(self.save_folder, 'best.pth')
                    torch.save(self.agent.state_dict(), savepath)
                    best_rew = cur_avg_rew
                    print('save')
            # self.update_manual()
            self.update_multiple()


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
        p_loss -= pi.entropy().mean()*self.entropy_kf

        v_loss = func.mse_loss(self.agent.value(states).flatten(), rtg)

        return p_loss, v_loss
    
    
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


    def update_multiple(self):

        p_loss_col, v_loss_col = np.zeros(0), np.zeros(0)

        states, act, rtg, adv, log_prob_old=self.buffer.get()
        b_inds = np.arange(len(rtg))

        for upd_i in range(self.optimization_iters):
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



    def load_best_weights(self, path=None):

        if path is None:
            path = os.path.join(
                    self.save_folder,
                    'best.pth'
                )

        self.agent.load_state_dict(
            torch.load(
                path
            )
        )