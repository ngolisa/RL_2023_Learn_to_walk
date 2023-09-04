import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()

        self.layer1 = nn.Conv2d(n_observations, 16, 4)
        self.layer15 = nn.MaxPool2d(2)
        self.layer2 = nn.Conv2d(16, 32, 3)
        self.layer3 = nn.MaxPool2d(2)
        self.layer4 = nn.Conv2d(32, 64, 3)
        self.layer5 = nn.MaxPool2d(2)
        self.layer6 = nn.Conv2d(64, 128, 2)
        self.layer7 = nn.MaxPool2d(2)
        self.layer8 = nn.Flatten()
        self.layer9 = nn.Linear(2048, 1024)
        self.layer95 = nn.Linear(1024, 256)
        self.layer10 = nn.Linear(256,5)


    def forward(self, states):
        """Forward pass."""
        x = states /255
        x = F.relu(self.layer1(x))
        x = self.layer15(x)
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        x = F.relu(self.layer4(x))
        x = self.layer5(x)
        x = F.relu(self.layer6(x))
        x = self.layer7(x)
        x = self.layer8(x)
        x = F.relu(self.layer9(x))
        x = F.relu(self.layer95(x))
        return F.softmax(self.layer10(x))
