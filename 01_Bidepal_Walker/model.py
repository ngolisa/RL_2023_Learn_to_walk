import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()

        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 256)
        self.layer3 = nn.Linear(256, 256)
        self.layer4 = nn.Linear(256, n_actions)

    def forward(self, states):
        """Forward pass."""
        x = F.relu(self.layer1(states))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return F.tanh(self.layer4(x))
