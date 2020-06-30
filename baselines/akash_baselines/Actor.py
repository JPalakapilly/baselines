import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, min_action, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        self.max_action = max_action
        self.min_action = min_action


    def forward(self, state, action):
        a = F.relu(self.l1(torch.cat([state, action], 1)))
        a = F.relu(self.l2(a))
        a = 0.05 * self.max_action * torch.tanh(self.l3(a))
        return (a + action).clamp(self.min_action, self.max_action)


