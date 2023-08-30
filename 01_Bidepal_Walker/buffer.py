import random
import torch
from collections import deque

from config import CFG

class ReplayBuffer():

    def __init__(self):
        self.buffer = deque(maxlen=CFG.buffer_size)

    def len(self):
        return len(self.buffer)

    def set(self, transi):
        """
        Store a single new transition in the replay buffer.
        """

        old, act, rwd, new, terminated = transi

        old = torch.tensor(old)
        act = torch.tensor(act)
        rwd = torch.tensor(rwd)
        new = torch.tensor(new)
        terminated = torch.tensor(terminated)

        # We deal with terminal states by giving them the same value as the previous state.
        # This ensures that CFG.gamma * V(new) - V(old) ~ 0
        if new is None:
            new = old

        # old = torch.permute(torch.tensor(old["observation"]).float(), (2, 0, 1))
        # new = torch.permute(torch.tensor(new["observation"]).float(), (2, 0, 1))
        self.buffer.append((old, act, rwd, new, terminated))

    def get(self):
        """
        Get a minibatch of observations of size CFG.batch_size.
        """
        if len(self.buffer) < CFG.batch_size:
            raise Exception("Not enough data in replay buffer.")

        batch = random.sample(self.buffer, CFG.batch_size)
        old, act, rwd, new, terminated = zip(*batch)

        old = torch.stack(old)
        new = torch.stack(new)
        act = torch.stack(act)
        rwd = torch.tensor(rwd).unsqueeze(-1)
        terminated = torch.tensor(terminated).unsqueeze(-1)

        return old, act, rwd, new, terminated

BUF = ReplayBuffer()
