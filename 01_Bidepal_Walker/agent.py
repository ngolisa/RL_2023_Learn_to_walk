"""
Agent module.
"""

import random
import torch
import torch.nn
import model
from config import CFG
from torch.nn import functional as F
import numpy as np


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
        self.start_expl = CFG.exploration_rate
        self.exploration_rate = CFG.exploration_rate
        self.exploration_rate_min = CFG.exploration_rate_min
        self.exploration_rate_decay = CFG.exploration_rate_decay


    def get(self, obs_new, act_space, evaluating=False):
        """
        Next action selection
        """
        # Evaluating agent means rely on learnt behavior (no randomness)
        if evaluating:
            self.exploration_rate=0.05


        # Explore : Returns a random action with p = exploration_rate
        if random.uniform(0,1) < self.exploration_rate:
            what_to_do = act_space.sample()

        # Exploit : Otherwise, returns the best choice according to the DQN
        else :
            with torch.no_grad():
                action = self.net(torch.tensor(obs_new))
                what_to_do = action.numpy()


        # decrease exploration_rate
        # self.exploration_rate = self.start_expl*(1-(self.time_step/(CFG.episodes*CFG.max_steps/4))**2)**.5

        #self.exploration_rate *= self.exploration_rate_decay

        self.exploration_rate = np.cos(self.time_step/(CFG.episodes*CFG.max_steps/3)*np.pi/2)
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        self.time_step += 1

        return what_to_do



    def set(self, obs_old, act, rwd, obs_new, terminated):
        """
        Learn from one step
        """


        # Get y_pred
        out = self.net(obs_old)

        terminated = terminated.long()


        # Get y_max from target
        with torch.no_grad():
            exp = rwd + (1 - terminated) * CFG.gamma * self.target(obs_new)

        # Compute loss
        loss = torch.abs(out - exp).mean()

        # Backward propagation
        # Gradient descent updates weight in the network to minimize loss
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
