import torch
from torch.optim import Adam

class Buffer:
    # folder for collecting stat-acttion pairs
    pass


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
                action, val = self.agent(state)
                state,rew,term,trunc,_ = self.env.step()
                self.buffer.collect(state, action, val, rew)

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

        
    def policy_loss(self, data):
        
        obs, act, adv, rew, log_prob_old = data 

        pi, log_prob = self.agent.policy(obs, act)
        ratio = torch.exp(log_prob - log_prob_old)
        ratio_cliped = torch.clamp(
            ratio, 1-self.clip_eps, 1+self.clip_eps)
        p_loss = -torch.min(ratio*adv, ratio_cliped*adv).mean()

        if self.agent.shared:
            p_loss += pi.entropy()

        v_loss = ((self.agent.value(obs) - rew)**2).mean()

        return p_loss, v_loss
    
