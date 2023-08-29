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
        self.target = model.DQN(obs_size, action_size)
        self.target.load_state_dict(self.net.state_dict())
        self.time_step = 0


    def get(self, obs_new, act_space, evaluating=False):
        """
        Next action selection
        """
        if evaluating:
            epsilon=0
        else:
            epsilon = CFG.epsilon

        # Returns a random action with p = epsilon or the best choice according to the DQN
        if random.uniform(0,1) < epsilon:
            return act_space.sample()

        with torch.no_grad():
            action = self.net(torch.tensor(obs_new))
            return action.numpy()


    def set(self, obs_old, act, rwd, obs_new):
        """
        Learn from one step
        """
        self.time_step += 1

        # Convert np.array from environment into tensor
        obs_old = torch.tensor(obs_old)
        obs_new = torch.tensor(obs_new)

        # Get y_pred
        out = self.net(obs_old)

        # Get y_max from target
        with torch.no_grad():
            exp = rwd + CFG.gamma * self.target(obs_new)

        # Compute loss
        loss = torch.square(exp-out).sum()

        # Backward propagation
        # Gradient descent updates weight in the network to minimize loss
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # Target network update every 'sync_every' steps
        if self.time_step % CFG.sync_every == 0:
            self.target.load_state_dict(self.net.state_dict())


    def save(self, path: str):
        """
        Save the agent's model to disk.
        """
        torch.save(self.net.state_dict(), f"{path}saved_model.pt")


    def load(self, path: str):
        """
        Load the agent's weights from disk.
        """
        data = torch.load(path, map_location=torch.device("cpu"))
        self.net.load_state_dict(data)
