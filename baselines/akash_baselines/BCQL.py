from Actor import Actor
from Critic import Critic
from VAE import VAE
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from utils import init_weights, copy_params, soft_update

class BCQ(object):
    def __init__(self, state_dim, action_dim, min_action, max_action, replay_buffer):

        latent_dim = action_dim * 2

        self.actor = Actor(state_dim, action_dim, min_action, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, min_action, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.vae = VAE(state_dim, action_dim, latent_dim, max_action).to(device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters()) 

        self.max_action = max_action
        self.min_action = min_action
        self.action_dim = action_dim
        
        self.replay_buffer = replay_buffer


    def select_action(self, state):		
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).repeat(10, 1).to(device)
            action = self.actor(state, self.vae.decode(state))
            q1 = self.critic.q1(state, action)
            ind = q1.max(0)[1]
        return action[ind].cpu().data.numpy().flatten()


    def train(self, iterations, batch_size=10, discount=0.99, tau=0.005):
        for it in range(iterations):
            print(str(it) + " / " + str(iterations))
            # Sample replay buffer / batch
            state_np,action, next_state_np, reward, done = self.replay_buffer.sample(batch_size)
            state 		= torch.FloatTensor(state_np).to(device)
            action 		= torch.FloatTensor(action).to(device)
            next_state 	= torch.FloatTensor(next_state_np).to(device)
            reward 		= torch.FloatTensor(reward).to(device)
            done 		= torch.FloatTensor(1 - done).to(device)


            # Variational Auto-Encoder Training
            recon, mean, std = self.vae(state, action)
            recon_loss = F.mse_loss(recon, action)
            KL_loss	= -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
            vae_loss = recon_loss + 0.5 * KL_loss

            self.vae_optimizer.zero_grad()
            vae_loss.backward()
            self.vae_optimizer.step()


            # Critic Training
            with torch.no_grad():

                # Duplicate state 10 times
                state_rep = torch.FloatTensor(np.repeat(next_state_np, 10, axis=0)).to(device)

                # Compute value of perturbed actions sampled from the VAE
                target_Q1, target_Q2 = self.critic_target(state_rep, self.actor_target(state_rep, self.vae.decode(state_rep)))

                # Soft Clipped Double Q-learning 
                target_Q = 0.75 * torch.min(target_Q1, target_Q2) + 0.25 * torch.max(target_Q1, target_Q2)
                target_Q = target_Q.view(batch_size, -1).max(1)[0].view(-1, 1)

                target_Q = reward + done * discount * target_Q

            current_Q1, current_Q2 = self.critic(state, action)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()


            # Pertubation Model / Action Training
            sampled_actions = self.vae.decode(state)
            perturbed_actions = self.actor(state, sampled_actions)

            # Update through DPG
            actor_loss = -self.critic.q1(state, perturbed_actions).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()


            # Update Target Networks 
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)