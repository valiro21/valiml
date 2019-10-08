import numpy as np

from valiml.rl.memory import CircularMemory
from valiml.rl.utils import SumSegmentTree


class PrioritizedMemory(CircularMemory):
    def __init__(self, memory_size=100000, alpha=1.0, eps=1e-7):
        super(PrioritizedMemory, self).__init__(memory_size=memory_size)
        self.sum_segment_tree = SumSegmentTree(memory_size)
        self.max_priority = 1
        self.eps = eps
        self.alpha = alpha

    def __len__(self):
        return self.n_items

    def add(self, experience):
        position = super(PrioritizedMemory, self).add(experience)
        priority = (self.max_priority + self.eps) ** self.alpha
        self.sum_segment_tree.update(position, priority)

    def priority_update(self, index, priority):
        self.max_priority = max(self.max_priority, priority)
        priority = (priority + self.eps) ** self.alpha
        self.sum_segment_tree.update(index, priority)

    def get_priority(self, index):
        return self.sum_segment_tree.get(index)

    def sample(self, sample_size=1, use_priority=True):
        if use_priority:
            return [self.sum_segment_tree.lower_bound(np.random.random()) for _ in range(sample_size)]
        return super(PrioritizedMemory, self).sample(sample_size=sample_size)
