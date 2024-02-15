import torch
from torch import nn

def linear(inp_shape, out_shape, act, use_bn):
    layer = [
        torch.nn.Linear(inp_shape, out_shape),
        act()
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

        self.stem = linear(inp_shape, hidden_shape, act, use_bn)
        self.stem.extend(
            linear(hidden_shape, hidden_shape, act, use_bn)*hidden_num
        )
        self.stem.append(linear(hidden_shape, out_shape, act, use_bn))
        self.stem = torch.nn.Sequential(self.stem)

    def forward(self, x):
        return self.stem(x)

                
class ActorCritic(torch.nn.Module):

    def __init__(self, obs_space, act_space):

        self.actor = MLP(obs_space, act_space)
