import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Re-tuned version of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971

class ConvModule(nn.Module):
	def __init__(self):
		super(ConvModule, self).__init__()

		self.conv1 = nn.Conv2d(3, 16, kernel_size=(2,2), stride=2)
		self.conv2 = nn.Conv2d(16, 16, kernel_size=(2,2), stride=2)
		self.conv3 = nn.Conv2d(16, 16, kernel_size=(2,2), stride=2)
		self.conv4 = nn.Conv2d(16, 16, kernel_size=(2,2), stride=2)


	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		x = F.relu(self.conv4(x))
		x = x.view((x.size(0), -1))
		return x


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 512)
		self.l2 = nn.Linear(512, 512)
		self.l3 = nn.Linear(512, action_dim)
		
		self.max_action = max_action


	def forward(self, x):
		x = F.relu(self.l1(x))
		x = F.relu(self.l2(x))
		x = self.max_action * torch.tanh(self.l3(x)) 
		return x 


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		self.l1 = nn.Linear(state_dim + action_dim, 512)
		self.l2 = nn.Linear(512, 512)
		self.l3 = nn.Linear(512, 1)


	def forward(self, x, u):
		x = F.relu(self.l1(torch.cat([x, u], 1)))
		x = F.relu(self.l2(x))
		x = self.l3(x)
		return x 


class DDPG(object):
	def __init__(self, state_dim, action_dim, max_action):
		if len(state_dim) > 1:
			# So as to implement a shared conv layer for actor and critic
			self.conv_module = ConvModule().to(device)
			state_dim = 16*5*5

		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target.load_state_dict(self.actor.state_dict())
		self.actor_optimizer = torch.optim.Adam(list(self.actor.parameters()) + list(self.conv_module.parameters()))

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = Critic(state_dim, action_dim).to(device)
		self.critic_target.load_state_dict(self.critic.state_dict())
		self.critic_optimizer = torch.optim.Adam(list(self.critic.parameters()) + list(self.conv_module.parameters()))


	def select_action(self, state):
		if len(state.shape) >= 3:
			state = torch.FloatTensor(state.reshape(1, *state.shape)).to(device)
			state = self.conv_module(state)
		else:
			state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()


	def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005):

		actor_losses = []
		critic_losses = []

		for it in range(iterations):

			# Sample replay buffer 
			x, y, u, r, d = replay_buffer.sample(batch_size)
			state = torch.FloatTensor(x).to(device)
			action = torch.FloatTensor(u).to(device)
			next_state = torch.FloatTensor(y).to(device)
			done = torch.FloatTensor(1 - d).to(device)
			reward = torch.FloatTensor(r).to(device)

			# Passing states through conv layers
			if len(state.shape) >= 3:
				state = self.conv_module(state)
				next_state = self.conv_module(next_state)

			# Compute the target Q value
			target_Q = self.critic_target(next_state, self.actor_target(next_state))
			target_Q = reward + (done * discount * target_Q).detach()

			# Get current Q estimate
			current_Q = self.critic(state, action)

			# Compute critic loss
			critic_loss = F.mse_loss(current_Q, target_Q)
			critic_losses.append(critic_loss)

			# Optimize the critic
			self.critic_optimizer.zero_grad()
			critic_loss.backward(retain_graph=True)
			self.critic_optimizer.step()

			# Compute actor loss
			actor_loss = -self.critic(state, self.actor(state)).mean()
			actor_losses.append(actor_loss)
			
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

		return sum(actor_losses)/iterations, sum(critic_losses)/iterations


	def save(self, filename, directory):
		torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
		torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))


	def load(self, filename, directory):
		self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
		self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
