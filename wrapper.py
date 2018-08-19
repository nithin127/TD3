import cv2
import math
import numpy as np
import gym
from gym import spaces

#avg_obs = np.load('im_avg_rand.npy')
avg_obs = np.load('im_avg_trial_run.npy')

class CustomWrap(gym.ObservationWrapper):
    def __init__(self, env=None, resize_w=80, resize_h=80):
        super().__init__(env)
        self.resize_h = resize_h
        self.resize_w = resize_w
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[1, 1, 1],
            [3, resize_h, resize_w],
            dtype=self.observation_space.dtype)

    def reset(self):
        obs = super().reset()
        # subtract mean
        obs = obs.astype(float) - avg_obs.transpose(2,1,0)
        # resize
        obs = cv2.resize(obs.swapaxes(0,2), dsize=(self.resize_w, self.resize_h), interpolation=cv2.INTER_CUBIC).swapaxes(0,2)
        # rescale
        obs /= 255.0
        # grayscale
        # obs = obs.mean(0)
        return obs

    def step(self, actions):
        obs, reward, done, info = super().step(actions)
        # subtract mean
        obs = obs.astype(float) - avg_obs.transpose(2,1,0)
        # resize
        obs = cv2.resize(obs.swapaxes(0,2), dsize=(self.resize_w, self.resize_h), interpolation=cv2.INTER_CUBIC).swapaxes(0,2)
        # rescale
        obs /= 255.0
        # grayscale
        # obs = obs.mean(0)
        # reward clipping
        reward = np.clip(reward, a_min=-10, a_max=1)
        return obs, reward, done, info

    def observation(self, observation):
        return observation.transpose(2, 1, 0)
