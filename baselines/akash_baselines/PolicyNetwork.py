import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import sys
sys.path.append("..")
from akash_baselines.utils import init_weights

torch.manual_seed(420)
class PolicyNetwork(nn.Module):
    def __init__(self,s_dim,a_dim,h_dim,action_space):
        super(PolicyNetwork,self).__init__()

        self.linear1 = nn.Linear(s_dim,max(h_dim,a_dim))
        self.linear2 = nn.Linear(h_dim,h_dim)

        self.linear3a = nn.Linear(max(h_dim,a_dim),a_dim)
        self.linear3b = nn.Linear(max(h_dim,a_dim),a_dim)

        # Apply weight initialisation to all linear layers
        self.apply(init_weights)

        # rescale actions
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self,s, min_log, max_log):
        x = F.relu(self.linear1(s))
        x = F.relu(self.linear2(x))
        mean = self.linear3a(x)
        log_std = self.linear3b(x)

        # constrain log value in finite range to avoid NaN loss values
        log_std = torch.clamp(log_std, min=min_log, max=max_log)
        
        return mean, log_std

    def sample_action(self,s, epsilon, min_log, max_log):
        mean, log_std = self.forward(s, min_log, max_log)
        std = log_std.exp()

        # calculate action using reparameterization trick and action scaling
        normal = Normal(mean, std)
        xi = normal.rsample()
        yi = torch.tanh(xi)
        a = yi * self.action_scale + self.action_bias
        log_pi = normal.log_prob(xi)

        # enforcing action bound (appendix of paper)
        log_pi -= torch.log(self.action_scale * (1 - yi.pow(2)) + epsilon)
        log_pi = log_pi.sum(1,keepdim=True)
        mean = torch.tanh(mean)*self.action_scale + self.action_bias

        return a, log_pi, mean