import numpy as np

from valiml.rl.memory import CircularMemory
from valiml.rl.utils import SumSegmentTree
from valiml.rl.utils.Treaps import get_kth, remove_by_value, insert
from valiml.rl.utils.RakingSampler import harmonic_sampler, harmonic_series_sum


class PrioritizedMemory(CircularMemory):
    def __init__(self, memory_size=100000, alpha=1.0, eps=1e-7, ranking=True):
        super(PrioritizedMemory, self).__init__(memory_size=memory_size)
        self.sum_segment_tree = SumSegmentTree(memory_size)
        self.treap = None
        self.max_priority = 1
        self.eps = eps
        self.alpha = alpha
        self.ranking = ranking
        self.value_counter = {}
        self.value_removed_counter = {}
        self.index_hash = {}

    def __len__(self):
        return self.n_items

    def _add(self, position, priority):
        if self.ranking:
            priority_value_count = self.value_counter.get(priority, 0)
            self.value_counter[priority] = priority_value_count + 1

            new_value = (priority, priority_value_count)
            self.index_hash[position] = new_value
            self.treap = insert(self.treap, np.random.random() * 1000 * self.memory_size, new_value, position)
        else:
            self.sum_segment_tree.update(position, priority)

    def add(self, experience):
        position = super(PrioritizedMemory, self).add(experience)
        priority = (self.max_priority + self.eps) ** self.alpha

        self._add(position, priority)

    def priority_update(self, index, priority):
        self.max_priority = max(self.max_priority, priority)
        priority = (priority + self.eps) ** self.alpha

        if self.ranking:
            treap_value = self.index_hash[index]
            old_priority = treap_value[0]
            self.treap = remove_by_value(self.treap, treap_value)

            n_removes = self.value_removed_counter.get(old_priority, 0)
            self.value_removed_counter[old_priority] = n_removes + 1

            if old_priority != priority and n_removes == self.value_counter[old_priority]:
                del self.value_counter[old_priority]
                del self.value_removed_counter[old_priority]

            self._add(index, priority)
        else:
            self.sum_segment_tree.update(index, priority)

    def get_normalized_priority(self, index):
        if self.ranking:
            return self.index_hash[index][0] / harmonic_series_sum(self.n_items)
        else:
            return self.sum_segment_tree.get(index) / self.sum_segment_tree.tree[1]

    def sample(self, sample_size=1, use_priority=True):
        if use_priority:
            if self.ranking:
                return [get_kth(self.treap, harmonic_sampler(self.n_items)).metadata for _ in range(sample_size)]
            else:
                return [self.sum_segment_tree.lower_bound(np.random.random()) for _ in range(sample_size)]
        return super(PrioritizedMemory, self).sample(sample_size=sample_size)
