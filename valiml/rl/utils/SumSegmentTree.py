import unittest


class SumSegmentTree(object):
    def __init__(self, size):
        self.size = size
        self.limit = 1
        while self.limit < size:
            self.limit *= 2

        self.tree = [0] * (2 * self.limit)

    def _update(self, index, start, end, position, value):
        if start == end:
            old_value = self.tree[index]
            self.tree[index] = value
            return old_value

        self.tree[index] += value

        middle = int((start + end) // 2)

        if position <= middle:
            old_value = self._update(2 * index, start, middle, position, value)
        else:
            old_value = self._update(2 * index + 1, middle + 1, end, position, value)

        self.tree[index] -= old_value
        return old_value

    def update(self, index, value):
        self._update(1, 1, self.limit, index + 1, value)

    def _get(self, index, start, end, position):
        if start == end:
            return self.tree[index]

        middle = int((start + end) // 2)
        if position <= middle:
            return self._get(2 * index, start, middle, position)
        else:
            return self._get(2 * index + 1, middle + 1, end, position)

    def get(self, index):
        return self._get(1, 1, self.limit, index + 1)

    def _lower_bound(self, index, start, end, cumulative_sum):
        if start == end:
            return start

        left_priority = self.tree[2 * index] / self.tree[1]
        middle = (start + end) // 2

        if cumulative_sum <= left_priority:
            return self._lower_bound(2 * index, start, middle, cumulative_sum)
        else:
            return self._lower_bound(2 * index + 1, middle + 1, end, cumulative_sum - left_priority)

    def lower_bound(self, cumulative_sum):
        return self._lower_bound(1, 1, self.limit, cumulative_sum) - 1


class SumSegmentTreeTests(unittest.TestCase):
    def test_update(self):
        tree = SumSegmentTree(1000)
        tree.update(1, 100)
        tree.update(2, 10)
        self.assertEqual(tree.get(1), 100)

    def test_lower_bound(self):
        tree = SumSegmentTree(1000)

        multiplier = 2.5
        tree.update(0, 0.2 * multiplier)
        tree.update(1, 0.5 * multiplier)
        tree.update(2, 0.1 * multiplier)
        tree.update(3, 0.05 * multiplier)
        tree.update(4, 0.05 * multiplier)
        tree.update(5, 0.1 * multiplier)

        self.assertEqual(tree.lower_bound(0.2), 0)
        self.assertEqual(tree.lower_bound(0.3), 1)
        self.assertEqual(tree.lower_bound(0.7), 1)
        self.assertEqual(tree.lower_bound(0.71), 2)


if __name__ == '__main__':
    unittest.main()
