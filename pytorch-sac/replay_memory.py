import pdb
import random
import numpy as np

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def get_stats(self):
        state, action, reward, next_state, done = map(np.stack, zip(*self.buffer))
        state_min = state.min(0)
        state_max = state.max(0)
        state_max[state_min==state_max] = state_min[state_min==state_max]+1
        #state_mean = state.mean(0)
        #state_std = state.std(0)
        return state_min, state_max#, state_mean, state_std

    def __len__(self):
        return len(self.buffer)
