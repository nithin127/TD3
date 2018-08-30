import numpy as np
import torch
import gym
import argparse
import os

import utils
import TD3
import OurDDPG
import DDPG

# Install https://github.com/duckietown/gym-duckietown/ before you start
import gym_duckietown
from wrapper import CustomWrap

# Logging facilities
from logger import Logger


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--policy_name", default="OurDDPG")				# Policy name
	parser.add_argument("--env_name", default="Duckietown-small_loop-v0")# OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)					# Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=1e4, type=int)		# How many time steps purely random policy is run for
	parser.add_argument("--eval_freq", default=5e3, type=float)			# How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=2e6, type=float)		# Max time steps to run environment for
	parser.add_argument("--avg-length", default=3, type=int)			# Actions are averaged over the following timesteps
	parser.add_argument("--obs-stack", default=3, type=int)				# Number of observations (current + previous) to be stacked, when given as input
	parser.add_argument("--expl_noise", default=0.1, type=float)		# Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=100, type=int)			# Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99, type=float)			# Discount factor
	parser.add_argument("--tau", default=0.005, type=float)				# Target network update rate
	parser.add_argument("--policy_noise", default=0.2, type=float)		# Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5, type=float)		# Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)			# Frequency of delayed policy updates
	args = parser.parse_args()
	# Rephrasing args
	
	file_name = "%s_%s_%s" % (args.policy_name, args.env_name, str(args.seed))
	print( "---------------------------------------")
	print( "Settings: %s" % (file_name))
	print( "---------------------------------------")

	env = CustomWrap(gym.make(args.env_name))

	# Set seeds
	env.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)

	state_dim = env.observation_space.shape # Assuming image as an input; otherwise .shape[0] would be appropriate
	action_dim = env.action_space.shape[0]
	max_action = int(env.action_space.high[0])

	# Initialize policy
	if args.policy_name == "TD3": policy = TD3.TD3(state_dim, action_dim, max_action)
	elif args.policy_name == "OurDDPG": policy = OurDDPG.DDPG(state_dim, action_dim, max_action)
	elif args.policy_name == "DDPG": policy = DDPG.DDPG(state_dim, action_dim, max_action)

	policy.load(file_name, directory="./pytorch_models")

	actions_test = np.zeros((args.avg_length, env.action_space.shape[0]))
	env.reset()
	env.render()
	while True:
		obs = env.reset()
		done = False
		total_reward = 0
		total_length = 0
		while not done:
			action = policy.select_action(np.array(obs))
			actions_test[1:args.avg_length] = actions_test[:args.avg_length - 1]
			actions_test[0] = action
			action = np.average(actions_test, 0)
			obs, reward, done, _ = env.step(action)
			print(action)
			env.render()
			total_reward += reward
			total_length += 1
			
	
		print( "---------------------------------------")
		print( "Episode reward: %f" % (total_reward) )
		print( "\tEpisode Length: %f" % (total_length) )
		print( "---------------------------------------")
	

