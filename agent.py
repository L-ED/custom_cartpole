import torch
from torch.distributions.categorical import Categorical

def linear(inp_shape, out_shape, act, use_bn):
    layer = [
        torch.nn.Linear(inp_shape, out_shape),
        # act()
    ]
    if use_bn:
        layer.insert(1, torch.nn.BatchNorm1d(out_shape))

    return layer


class MLP(torch.nn.Module):

    def __init__(
            self, 
            inp_shape, 
            out_shape, 
            hidden_shape=64, 
            hidden_num=1, 
            act= torch.nn.ReLU, 
            use_bn = False
        ):
        super().__init__()

        stem = linear(inp_shape, hidden_shape, act, use_bn)
        stem.extend(
            linear(hidden_shape, hidden_shape, act, use_bn)*hidden_num
        )
        stem.extend(linear(hidden_shape, out_shape, act, use_bn))
        self.stem = torch.nn.Sequential(*stem)


    def forward(self, x):
        return self.stem(x)

                
class ActorCritic(torch.nn.Module):

    def __init__(self, obs_space, act_space):
        super().__init__()

        self.shared = False
        obs_len= obs_space.shape[0]
        self.policy = MLP(obs_len, act_space.n)
        self.value = MLP(obs_len, 1)


    def get_probs(self, obs, act):
        distrib = Categorical(
            logits=self.policy(obs))
        log_prob = distrib.log_prob(act)
        return distrib, log_prob


    def step(self, obs):
        distrib = Categorical(
            logits=self.policy(obs))
        act = distrib.sample()
        log_prob = distrib.log_prob(act)
        value = self.value(obs).squeeze(-1)
        return act, value, log_prob