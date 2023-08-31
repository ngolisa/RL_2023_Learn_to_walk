"""
Agent module.
"""

import random
import torch
import torch.nn
import model
import gym
from config import CFG
from torch.nn import functional as F
from gym.utils.save_video import save_video



class Agent:
    """
    A learning agent parent class.
    """

    def __init__(self):
        pass

    def set(self):
        """
        Make the agent learn from a (s, a, r, s') tuple.
        """
        raise NotImplementedError

    def get(self):
        """
        Request a next action from the agent.
        """
        raise NotImplementedError

    def save(self, path: str, best_reward):
        """
        Save the agent's model to disk.
        """
        torch.save(self.net.state_dict(), f"{path}saved_model_{type(self).__name__}__{round(best_reward,2)}rw__{CFG.episodes}episodes__{CFG.max_steps}steps.pt")


    def load(self, path: str):
        """
        Load the agent's weights from disk.
        """
        data = torch.load(path, map_location=torch.device("cpu"))
        self.net.load_state_dict(data)

    def evaluate(self, number_of_steps=1000):
        input("Press enter to see validation ")
        env = gym.make("BipedalWalker-v3",hardcore=False, render_mode='human')
        obs_old, info = env.reset()
        terminated = False

        for step in range(number_of_steps):
            if terminated :
                obs_old, info = env.reset()
            action = self.get(obs_old, env.action_space, evaluating=True)
            obs_new, reward, terminated, truncated, info = env.step(action)

        obs_old = obs_new


    # def evaluate_recording(self, number_of_steps=1000, frequency_episodes=1):
    #     env = gym.make("BipedalWalker-v3",hardcore=False, render_mode='rgb_array')
    #     input("Press enter to see validation ")
    #     obs_old, info = env.reset()
    #     terminated = False
    #     episode_index = 0
    #     step_starting_index =0

    #     for step in range(1000):
    #         if terminated :

    #             episode_index += 1
    #             step_starting_index = step+1

    #             obs_old, info = env.reset()
    #         if episode_index%frequency_episodes==0:
    #             save_video(
    #                 env.render(),
    #                 "videos-bidepal",
    #                 fps=env.metadata["render_fps"],
    #                 step_starting_index=step_starting_index,
    #                 episode_index=episode_index,
    #                 name_prefix = f"{type(self).__name__}__{CFG.episodes}episodes__{CFG.max_steps}steps__"
    #             )


    #         action = self.get(obs_old, env.action_space, evaluating=True)

    #         obs_new, reward, terminated, truncated, info = env.step(action)

    #         obs_old = obs_new





class RandomAgent(Agent):
    """
    A random playing agent class.
    """

    def set(self, obs_old, act, rwd, obs_new):
        """
        A random agent doesn't learn.
        """
        return

    def get(self, obs_new, act_space):
        """
        Simply return a random action.
        """
        return act_space.sample()

class DQNAgent(Agent):
    """
    A basic DQN agent
    """
    def __init__(self, obs_size, action_size):
        super().__init__()
        """
        Initializing model and parameters
        """
        self.net = model.DQN(obs_size, action_size)
        self.opt = torch.optim.Adam(params=self.net.parameters(), lr=CFG.learning_rate)
        self.target = model.DQN(obs_size, action_size)
        self.target.load_state_dict(self.net.state_dict())
        self.time_step = 0
        self.exploration_rate = CFG.exploration_rate
        self.exploration_rate_min = CFG.exploration_rate_min
        self.exploration_rate_decay = CFG.exploration_rate_decay


    def get(self, obs_new, act_space, evaluating=False):
        """
        Next action selection
        """
        # Evaluating agent means rely on learnt behavior (no randomness)
        if evaluating:
            self.exploration_rate=0

        # Explore : Returns a random action with p = exploration_rate
        if random.uniform(0,1) < self.exploration_rate:
            what_to_do = act_space.sample()

        # Exploit : Otherwise, returns the best choice according to the DQN
        else :
            with torch.no_grad():
                action = self.net(torch.tensor(obs_new))
                what_to_do = action.numpy()


        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        return what_to_do



    def set(self, obs_old, act, rwd, obs_new, terminated):
        """
        Learn from one step
        """
        self.time_step += 1

        # Get y_pred
        out = self.net(obs_old)

        terminated = terminated.long()


        # Get y_max from target
        with torch.no_grad():
            exp = rwd + (1 - terminated) * CFG.gamma * self.target(obs_new)

        # Compute loss
        loss = torch.square(out - exp)


        # Backward propagation
        # Gradient descent updates weight in the network to minimize loss
        self.opt.zero_grad()
        loss.sum().backward()
        self.opt.step()

        # Target network update every 'sync_every' steps
        if self.time_step % CFG.sync_every == 0:
            self.target.load_state_dict(self.net.state_dict())
