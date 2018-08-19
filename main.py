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


# Runs policy for X episodes and returns average reward
def evaluate_policy(policy, eval_episodes=10, avg_len=3):
	avg_reward = 0.
	actions_test = np.zeros((avg_len, env.action_space.shape[0]))
	for _ in range(eval_episodes):
		obs = env.reset()
		done = False
		while not done:
			action = policy.select_action(np.array(obs))
			actions_test[1:avg_len] = actions_test[:avg_len - 1]
			actions_test[0] = action
			action = np.average(actions_test, 0)
			obs, reward, done, _ = env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes

	print( "---------------------------------------")
	print( "Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
	print( "---------------------------------------")
	return avg_reward


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
	parser.add_argument("--dont-save-model", action="store_true")			# Whether or not models are saved
	parser.add_argument("--expl_noise", default=0.1, type=float)		# Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=100, type=int)			# Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99, type=float)			# Discount factor
	parser.add_argument("--tau", default=0.005, type=float)				# Target network update rate
	parser.add_argument("--policy_noise", default=0.2, type=float)		# Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5, type=float)		# Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)			# Frequency of delayed policy updates
	args = parser.parse_args()
	# Rephrasing args
	args.save_models = not args.dont_save_model

	file_name = "%s_%s_%s" % (args.policy_name, args.env_name, str(args.seed))
	print( "---------------------------------------")
	print( "Settings: %s" % (file_name))
	print( "---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")
	if args.save_models and not os.path.exists("./pytorch_models"):
		os.makedirs("./pytorch_models")
	logger_dir = os.path.join("./results", file_name)
	logger = Logger(logger_dir)

	env = CustomWrap(gym.make(args.env_name))

	# Set seeds
	env.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)

	state_dim = env.observation_space.shape #Assuming image as an input; otherwise .shape[0] would be appropriate
	action_dim = env.action_space.shape[0]
	max_action = int(env.action_space.high[0])

	# Initialize policy
	if args.policy_name == "TD3": policy = TD3.TD3(state_dim, action_dim, max_action)
	elif args.policy_name == "OurDDPG": policy = OurDDPG.DDPG(state_dim, action_dim, max_action)
	elif args.policy_name == "DDPG": policy = DDPG.DDPG(state_dim, action_dim, max_action)

	replay_buffer = utils.ReplayBuffer()

	total_timesteps = 0
	timesteps_since_eval = 0
	episode_num = 0
	num_updates = 0
	done = True
	rewards = []
	actions = []

	# Evaluate untrained policy
	test_reward = evaluate_policy(policy, avg_len = args.avg_length)
	logger.log_scalar_rl("test_reward", test_reward, [episode_num, total_timesteps, num_updates])
	evaluations = [test_reward]
	
	while total_timesteps < args.max_timesteps:

		if done: 

			if total_timesteps != 0: 
				print("Total T: %d Episode Num: %d Episode T: %d Reward: %f" % (total_timesteps, episode_num, episode_timesteps, episode_reward))
				num_updates += episode_timesteps
				if args.policy_name == "TD3":
					actor_loss, critic_loss = policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau, args.policy_noise, args.noise_clip, args.policy_freq)
				else: 
					actor_loss, critic_loss = policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau)
				logger.log_scalar_rl("actor_loss", actor_loss, [episode_num, total_timesteps, num_updates])
				logger.log_scalar_rl("critic_loss", critic_loss, [episode_num, total_timesteps, num_updates])


			# Evaluate episode
			if timesteps_since_eval >= args.eval_freq:
				timesteps_since_eval %= args.eval_freq
				test_reward = evaluate_policy(policy, avg_len = args.avg_length)
				evaluations.append(test_reward)
				# Log rewards, actions 
				logger.log_scalar_rl("test_reward", test_reward, [episode_num, total_timesteps, num_updates])
				logger.log_histogram("train_actions_hist", actions, num_updates)
				logger.log_scalar_rl("train_avg_reward", sum(rewards)/len(rewards), [episode_num, total_timesteps, num_updates])
				actions = []
				rewards = []

	
				if args.save_models: policy.save(file_name, directory="./pytorch_models")
				np.save("./results/%s/evaluations" % (file_name), evaluations) 

			# Reset environment
			obs = env.reset()
			done = False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1 

		actions_train = np.zeros((args.avg_length, action_dim))
		# Select action randomly or according to policy
		if total_timesteps < args.start_timesteps:
			action = env.action_space.sample()
		else:
			action = policy.select_action(np.array(obs))
			actions_train[1:args.avg_length] = actions_train[:args.avg_length - 1]
			actions_train[0] = action
			action = np.average(actions_train, 0)
			if args.expl_noise != 0: 
				action = (action + np.random.normal(0, args.expl_noise, size=env.action_space.shape[0])).clip(env.action_space.low, env.action_space.high)


		# Perform action
		new_obs, reward, done, _ = env.step(action) 
		actions.append(action)
		rewards.append(reward)
		#done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)
		done_bool = float(done)
		episode_reward += reward

		# Store data in replay buffer
		replay_buffer.add((obs, new_obs, action, reward, done_bool))

		obs = new_obs

		episode_timesteps += 1
		total_timesteps += 1
		timesteps_since_eval += 1

	# Final evaluation 
	test_reward = evaluate_policy(policy, avg_len = args.avg_length)
	logger.log_scalar_rl("test_reward", test_reward, [episode_num, total_timesteps, num_updates])
	evaluations.append(test_reward)
	if args.save_models: policy.save("%s" % (file_name), directory="./pytorch_models")
	np.save("./results/%s/evaluations" % (file_name), evaluations)
