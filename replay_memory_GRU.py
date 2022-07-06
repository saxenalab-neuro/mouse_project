import random
import numpy as np
from itertools import chain
import torch

class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done, h_current):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done, h_current)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, h_current = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done, h_current

    def __len__(self):
        return len(self.buffer)

class PolicyReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = state
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        state = [[list(element)[0] for element in sample]for sample in batch]
        state = list(map(torch.FloatTensor, state))

        action = [[list(element)[1] for element in sample]for sample in batch]
        action = list(map(torch.FloatTensor, action))

        reward = [[list(element)[2] for element in sample]for sample in batch]
        reward = list(map(torch.FloatTensor, reward))

        next_state = [[list(element)[3] for element in sample]for sample in batch]
        next_state = list(map(torch.FloatTensor, next_state))

        done = [[list(element)[4] for element in sample]for sample in batch]
        done = list(map(torch.FloatTensor, done))

        hi_lst = []
        ho_lst = []

        for sample in batch:
            hi_lst.append(list(sample[0])[5])
            ho_lst.append(list(sample[0])[6])

        hi_lst = torch.cat(hi_lst, dim= -2).detach()
        ho_lst = torch.cat(ho_lst, dim= -2).detach()

        hidden_in = (hi_lst)
        hidden_out = (ho_lst)       


        return state, action, reward, next_state, done, hidden_in, hidden_out

    def __len__(self):
        return len(self.buffer)
