"""
Agent module.
"""

import random
import torch
import torch.nn
import model
from config import CFG


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


    def set(self, obs_old, act, rwd, obs_new):
        """
        Learn from one step
        """
        obs_old = torch.tensor(obs_old)
        obs_new = torch.tensor(obs_new)

        out = self.net(obs_old)

        with torch.no_grad():
            exp = rwd + CFG.gamma * self.net(obs_new)

        #loss
        loss = torch.square(exp-out).sum()

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()



    def get(self, obs_new, act_space):
        """
        Returns the best action according to the DQN.
        """
        epsilon = CFG.epsilon

        # Returns a random action with p = epsilon or the best choice according to the nn
        if random.uniform(0,1) < epsilon:
            return act_space.sample()

        with torch.no_grad():
            action = self.net(torch.tensor(obs_new))
            return action.numpy()
