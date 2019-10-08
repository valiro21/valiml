import numpy as np


class CircularMemory(object):
    def __init__(self, memory_size=1000000):
        self.memory_size = memory_size
        self.memory = [None] * memory_size
        self.write_head = 0
        self.n_items = 0

    def __len__(self):
        return self.n_items

    def add(self, experience):
        position = self.write_head
        self.memory[position] = experience

        self.write_head = (self.write_head + 1) % self.memory_size
        self.n_items = min(self.n_items + 1, self.memory_size)

        return position

    def sample(self, sample_size=1):
        return np.random.choice(self.n_items, size=sample_size, replace=False)

    def __getitem__(self, item):
        if isinstance(item, list):
            return [self.memory[idx] for idx in item]
        return self.memory[item]
