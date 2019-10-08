import unittest


class TreapNode(object):
    def __init__(self, priority, value, left_treap, right_treap, metadata=None):
        self.left = left_treap
        self.right = right_treap
        self.value = value
        self.priority = priority
        self.metadata = metadata
        self.n_items = self.compute_n_items()

    def compute_n_items(self):
        return 1 + \
               (0 if self.left is None else self.left.n_items) + \
               (0 if self.right is None else self.right.n_items)

    def set_left_node(self, treap_node):
        self.left = treap_node
        self.n_items = self.compute_n_items()

    def set_right_node(self, treap_node):
        self.right = treap_node
        self.n_items = self.compute_n_items()


class TreapNodePtr(object):
    def __init__(self, treap_node):
        self.treap_node = treap_node

    def set_treap_node(self, treap_node):
        self.treap_node = treap_node

    def get_treap_node(self):
        return self.treap_node


def merge(treap_node_ptr, left, right):
    if left is None or right is None:
        treap_node_ptr.set_treap_node(left if left is not None else right)
    elif left.priority > right.priority:
        new_treap_node_prt = TreapNodePtr(None)
        merge(new_treap_node_prt, left.right, right)
        left.set_right_node(new_treap_node_prt.get_treap_node())
        treap_node_ptr.set_treap_node(left)
    else:
        new_treap_node_ptr = TreapNodePtr(None)
        merge(new_treap_node_ptr, left, right.left)
        right.set_left_node(new_treap_node_ptr.get_treap_node())
        treap_node_ptr.set_treap_node(right)


def split(treap, value, left_treap_node_ptr, right_treap_node_ptr):
    if treap is None:
        left_treap_node_ptr.set_treap_node(None)
        right_treap_node_ptr.set_treap_node(None)
    elif value <= treap.value:
        new_treap_node_ptr = TreapNodePtr(None)
        split(treap.left, value, left_treap_node_ptr, new_treap_node_ptr)
        treap.set_left_node(new_treap_node_ptr.get_treap_node())
        right_treap_node_ptr.set_treap_node(treap)
    else:
        new_treap_node_ptr = TreapNodePtr(None)
        split(treap.right, value, new_treap_node_ptr, right_treap_node_ptr)
        treap.set_right_node(new_treap_node_ptr.get_treap_node())
        left_treap_node_ptr.set_treap_node(treap)


def insert(treap, priority, value, metadata=None):
    if treap is None:
        return TreapNode(priority, value, None, None, metadata=metadata)

    left_treap_node_ptr = TreapNodePtr(None)
    right_treap_node_ptr = TreapNodePtr(None)

    split(treap, value, left_treap_node_ptr, right_treap_node_ptr)

    merged_treap = TreapNodePtr(None)
    merge(merged_treap, left_treap_node_ptr.get_treap_node(), TreapNode(priority, value, None, None, metadata=metadata))
    merge(merged_treap, merged_treap.get_treap_node(), right_treap_node_ptr.get_treap_node())
    return merged_treap.get_treap_node()


def remove_kth(treap, k):
    left_items = 0 if treap.left is None else treap.left.n_items

    if left_items == k:
        new_treap_node_ptr = TreapNodePtr(None)
        merge(new_treap_node_ptr, treap.left, treap.right)
        return new_treap_node_ptr.get_treap_node()
    elif k <= left_items:
        new_node = remove_kth(treap.left, k)
        treap.set_left_node(new_node)
    else:
        new_node = remove_kth(treap.right, k - left_items - 1)
        treap.set_right_node(new_node)
    return treap


def remove_by_value(treap, value):
    if value < treap.value:
        new_node = remove_by_value(treap.left, value)
        treap.set_left_node(new_node)
    elif treap.value < value:
        new_node = remove_by_value(treap.right, value)
        treap.set_right_node(new_node)
    else:
        new_treap_node_ptr = TreapNodePtr(None)
        merge(new_treap_node_ptr, treap.left, treap.right)
        return new_treap_node_ptr.get_treap_node()
    return treap


def get_kth(treap, k):
    left_items = 0 if treap.left is None else treap.left.n_items

    if left_items == k:
        return treap
    elif k <= left_items:
        return get_kth(treap.left, k)
    else:
        return get_kth(treap.right, k - left_items - 1)


def create_treap(priority, value):
    return TreapNode(priority, value, None, None)


class Tests(unittest.TestCase):
    #TODO: Tests should be split
    def test_treap(self):
        #TEST INSERT
        treap = None
        treap = insert(treap, 5, 10)
        treap = insert(treap, 3, 8)
        treap = insert(treap, 2, 14)
        treap = insert(treap, 16, 15)

        self.assertEqual(treap.n_items, 4)
        self.assertEqual(get_kth(treap, 0).value, 8)
        self.assertEqual(get_kth(treap, 1).value, 10)
        self.assertEqual(get_kth(treap, 2).value, 14)

        # TEST REMOVE
        treap = remove_by_value(treap, 10)
        self.assertEqual(treap.n_items, 3)
        self.assertEqual(get_kth(treap, 0).value, 8)

        treap = remove_by_value(treap, 15)
        self.assertEqual(treap.n_items, 2)
        self.assertEqual(get_kth(treap, 0).value, 8)

        treap = remove_by_value(treap, 8)
        self.assertEqual(treap.n_items, 1)
        self.assertEqual(get_kth(treap, 0).value, 14)

        treap = remove_by_value(treap, 14)
        self.assertEqual(treap, None)


if __name__ == '__main__':
    unittest.main()
