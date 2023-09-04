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

        old = torch.tensor(old).to(CFG.device)
        act = torch.tensor(act).to(CFG.device)
        rwd = torch.tensor(rwd).to(CFG.device)
        new = torch.tensor(new).to(CFG.device)
        terminated = torch.tensor(terminated).to(CFG.device)


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
        rwd = torch.tensor(rwd).to(CFG.device) #.unsqueeze(-1)
        terminated = torch.tensor(terminated).to(CFG.device) # .unsqueeze(-1)

        return old, act, rwd, new, terminated

    def clear(self):
        """
        Clear the buffer memory
        """
        self.buffer.clear()


BUF = ReplayBuffer()
