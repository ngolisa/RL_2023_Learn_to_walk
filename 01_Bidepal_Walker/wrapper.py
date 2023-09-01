import gym
import numpy as np

class RewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # if reward>0:
        #     reward+=100

        # if obs[8] + obs[13] == 1:
        #     reward += 2

        # if np.abs(obs[0]) <= .5:
        #     reward += 1

        # if obs[2]>.3:
        #     reward+=100

        # if max(obs[7]/obs[12], obs[7]/obs[12])  <= 5:
        #     reward -=1
        # # else :
        # #     reward -= 1
        return obs, reward, terminated, truncated, info
