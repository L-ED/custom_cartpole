import torch, math
from torch.distributions.categorical import Categorical

def linear(inp_shape, out_shape, act, use_bn, init_fn=None):
    
    lin = torch.nn.Linear(inp_shape, out_shape)
    if init_fn is not None:
        lin = init_fn(lin, act)
    layer = [
        lin,
        act()
    ]
    if use_bn:
        layer.insert(1, torch.nn.BatchNorm1d(out_shape))

    return layer


def orthogonal_init(layer, act, std=math.sqrt(2), bias_const=0.0, last=False):
    if last:
        std = 0.01
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def glorot_init(layer, act, bias_const=0.0, last=False):
    gain = torch.nn.init.calculate_gain(acts[act])
    if last:
        gain /=100
    torch.nn.init.xavier_uniform_(layer.weight, gain)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


initialisers = {
    'glorot': glorot_init, 'ortho':orthogonal_init
}
acts = {
    torch.nn.Tanh:'tanh',
    torch.nn.ReLU:'relu',
    torch.nn.SELU:'linear', #  Self-Normalizing Neural Networks
    torch.nn.LeakyReLU:'leaky_relu'
}


class MLP(torch.nn.Module):

    def __init__(
            self, 
            inp_shape, 
            out_shape, 
            hidden_shape=64, 
            hidden_num=2, 
            act= torch.nn.Tanh, 
            use_bn = False,
            weight_init = 'glorot'#'ortho'#'glorot'
        ):
        super().__init__()

        init_fn =None
        if weight_init is not None:
            init_fn = initialisers[weight_init]

        stem = linear(inp_shape, hidden_shape, act, use_bn, init_fn=init_fn)
        stem.extend(
            linear(hidden_shape, hidden_shape, act, use_bn, init_fn=init_fn)*hidden_num
        )

        last = torch.nn.Linear(hidden_shape, out_shape)
        if init_fn is not None:
            last = init_fn(
                last,
                act,
                last=True
            )
        stem.append(last)
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

    def act(self, obs):
        with torch.no_grad():
            return torch.argmax(self.policy(obs), dim=1)

    def step(self, obs):
        with torch.no_grad():
            distrib = Categorical(
                logits=self.policy(obs))
            act = distrib.sample()
            log_prob = distrib.log_prob(act).squeeze(-1)
            value = self.value(obs).squeeze(-1)
        return act, value, log_prob