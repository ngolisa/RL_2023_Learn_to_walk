

import gym
from gym import RewardWrapper


class RewardWrapper(gym.RewardWrapper):
    def __init__(self, env, obs):
        super().__init__(env)
        self.obs = obs


    def reward(self, reward):
        # Scale rewards:

        reward =self.obs[0]
        return reward
        pass

env = gym.make("BipedalWalker-v3",hardcore=False, render_mode='human')
obs_new,_ = env.reset()
env = RewardWrapper(env, obs_new)

for step in range(10):


    print(obs_new)

    action = env.action_space.sample()

    obs_new, reward_new, terminated_new, truncated_new, info_new = env.step(action)
    # env.obs_new=obs_new
    env = RewardWrapper(env, obs_new)

    print(reward_new)
    print('___')


# obs_new,_ = new_env.reset()

# obs_new_new, reward_new, terminated_new, truncated_new, info_new = new_env.step(action)
# print(reward_new)

# print('______')

# print(env.observation_space)

# breakpoint()

# obs_new,_ = env.reset()
# # print(obs_new)
# obs_new, reward, terminated, truncated, info = env.step(action)
# # print(reward)
