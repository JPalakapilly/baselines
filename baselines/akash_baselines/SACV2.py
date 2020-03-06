import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
import torch.distributions as tdist
import torch.optim as opt

from utils import init_weights, copy_params, soft_update
from QNetwork import QNetwork
from PolicyNetwork import PolicyNetwork

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TAU = 0.005
EPSILON = 1e-6
H_DIM = 32
LR = 3e-4
REPLAY_MEMORY_SIZE = 50000
ALPHA = 1.0
GAMMA = 0.99 # 0.98
ENTROPY_TUNING = True # True
MIN_LOG = -20
MAX_LOG = 2

class SoftActorCritic(object):
    def __init__(self,observation_space,action_space, memory):
        self.s_dim = observation_space.shape[0]
        self.a_dim = action_space.shape[0]
        self.alpha = ALPHA

        # create component networks
        self.q_network_1 = QNetwork(self.s_dim,self.a_dim,H_DIM).to(device)
        self.q_network_2 = QNetwork(self.s_dim,self.a_dim,H_DIM).to(device)
        self.target_q_network_1 = QNetwork(self.s_dim,self.a_dim,H_DIM).to(device)
        self.target_q_network_2 = QNetwork(self.s_dim,self.a_dim,H_DIM).to(device)
        self.policy_network = PolicyNetwork(self.s_dim, self.a_dim, H_DIM, action_space).to(device)

        # copy weights from q networks to target networks
        copy_params(self.target_q_network_1, self.q_network_1)
        copy_params(self.target_q_network_2, self.q_network_2)
        
        # optimizers
        self.q_network_1_opt = opt.Adam(self.q_network_1.parameters(),LR)
        self.q_network_2_opt = opt.Adam(self.q_network_2.parameters(),LR)
        self.policy_network_opt = opt.Adam(self.policy_network.parameters(),LR)
        
        # automatic entropy tuning
        if ENTROPY_TUNING:
            self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optim = opt.Adam([self.log_alpha], lr=LR)
                
        self.replay_memory = memory

    def get_action(self, s):
        state = torch.FloatTensor(s).to(device).unsqueeze(0)
        action, _, _ = self.policy_network.sample_action(state, EPSILON, MIN_LOG, MAX_LOG)
        return action.detach().cpu().numpy()[0]
    
#     def overfit_update_params(self,batch_size, q1_prev_loss, q2_prev_loss, policy_prev_loss, alpha_prev_loss):
#         if(q1_prev_loss == None):
#             return (self.update_params(batch_size),1)
#         else:
#             num_iters = 1
            
#             #Stupid hack b/c this overfitting keeps switching minima
# #             if(q1_prev_loss <= 0.05 and q2_prev_loss <= 0.05):
# #                 return ((q1_prev_loss, q2_prev_loss, policy_prev_loss, alpha_prev_loss), 0)
            
#             q1_loss, q2_loss, policy_loss, alpha_loss = self.update_params(batch_size)
#             while(num_iters <= 100):
#                 num_iters += 1
#                 if(q1_loss < q1_prev_loss and q2_loss < q2_prev_loss and policy_loss <= policy_prev_loss):
#                     return ((q1_loss, q2_loss, policy_loss, alpha_loss), num_iters)
#                 else:
#                     q1_loss, q2_loss, policy_loss, alpha_loss = self.update_params(batch_size)
#             return ((q1_loss, q2_loss, policy_loss, alpha_loss), num_iters)
        
    def update_params(self, batch_size):
        
        states, actions, rewards, next_states, ndones = self.replay_memory.sample(batch_size)
        
        # make sure all are torch tensors
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        ndones = torch.FloatTensor(np.float32(ndones)).unsqueeze(1).to(device)

        # compute targets
        with torch.no_grad():
            next_action, next_log_pi,_ = self.policy_network.sample_action(next_states, EPSILON, MIN_LOG, MAX_LOG)
            next_target_q1 = self.target_q_network_1(next_states,next_action)
            next_target_q2 = self.target_q_network_2(next_states,next_action)
            next_target_q = torch.min(next_target_q1,next_target_q2) - self.alpha*next_log_pi
            next_q = rewards + GAMMA*next_target_q

        # compute losses
        q1 = self.q_network_1(states,actions)
        q2 = self.q_network_2(states,actions)

        q1_loss = F.mse_loss(q1,next_q)
        q2_loss = F.mse_loss(q2,next_q)
        
        pi, log_pi,_ = self.policy_network.sample_action(states, EPSILON, MIN_LOG, MAX_LOG)
        q1_pi = self.q_network_1(states,pi)
        q2_pi = self.q_network_2(states,pi)
        min_q_pi = torch.min(q1_pi,q2_pi)

        policy_loss = ((self.alpha * log_pi) - min_q_pi).mean()
        

        # gradient descent
        self.q_network_1_opt.zero_grad()
        q1_loss.backward()
        self.q_network_1_opt.step()

        self.q_network_2_opt.zero_grad()
        q2_loss.backward()
        self.q_network_2_opt.step()

        self.policy_network_opt.zero_grad()
        policy_loss.backward()
        self.policy_network_opt.step()

        # alpha loss
        if ENTROPY_TUNING:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
        else:
            alpha_loss = torch.tensor(0.).to(device)

        # update target network params
        soft_update(self.target_q_network_1,self.q_network_1, TAU)
        soft_update(self.target_q_network_2,self.q_network_2, TAU)

        return q1_loss.item(), q2_loss.item(), policy_loss.item(), alpha_loss.item()