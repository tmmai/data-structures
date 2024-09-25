from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, Optional, Sequence, TypeVar
import unittest


class Comparable(metaclass=ABCMeta):
    @abstractmethod
    def __lt__(self, other: Any) -> bool: ...


K = TypeVar('K', bound=Comparable)


@dataclass
class AVLNode(Generic[K]):
    key: K
    height: int = field(default = 0)

    # Children represented as [left_child, right_child]
    children: list[Optional['AVLNode[K]']] = field(default_factory=lambda: [None, None])


class Direction(Enum):
    """Signifies an AVL rotation direction."""
    LEFT: int = 0
    RIGHT: int = 1


class AVLTree():
    """An implementaetion of an AVL Tree.

    An AVL tree is self-balancing binary search tree (BST) that maintains
    the following invariants:
    For every node n...
        - n's left subtree is < n's key.
        - n's right subtree is > n's key.
        - n has 0, 1, or 2 children.
        - n's left subtree and right subtree differ in height by no more than 1.
    
    Typical usage example:
        avl_tree = AVLTree()
        avl_tree.insert(2)
        avl_tree.insert(1)
        avl_tree.insert(3)
        avl_tree.contains(1)
        avl_tree.remove(3)
        avl_tree.get_sorted_keys()

        avl_tree = AVLTree([3, 5, 7, 2, 9])
    """
    def __init__(self, sequence: Optional[Sequence[K]] = None):
        self._root: Optional['AVLNode[K]'] = None
        self._keys: set = set()
        if sequence:
            for key in sequence: 
                self.insert(key)            
    
    @property
    def size(self) -> int:
        """Returns the number of keys in the AVL tree."""
        return len(self._keys)

    def insert(self, key: K):
        """Inserts the given key into the AVL tree."""
        if key in self._keys: return

        self._root = self._insert_node(self._root, key)
        self._keys.add(key)

    def remove(self, key: K):
        """Removes the given key from the AVL tree if it exists."""
        if key not in self._keys: return

        self._root = self._remove(self._root, key)
        self._keys.remove(key)

    def contains(self, key: K) -> bool:
        """Returns true if the given key is in the AVL tree, false else."""
        return key in self._keys
    
    def get_sorted_keys(self) -> Sequence[K]:
        """Returns a sequence of the keys existing in the AVL tree in sorted order."""
        sorted_keys = []
        def inorder(node: Optional['AVLNode[K]']):
            if not node: return

            inorder(node.children[0])
            sorted_keys.append(node.key)
            inorder(node.children[1])
        
        inorder(self._root)
        return sorted_keys

    def _get(self, root: Optional['AVLNode[K]'], key: K) -> Optional['AVLNode[K]']:
        """Gets the AVLNode belonging to the given key.

        Args:
            root: The root of the subtree to search.
            key: The desired key of the node.
        """
        if not root: return None
        elif root.key == key: return root

        index = 0 if key < root.key else 1
        return self._get(root.children[index], key)

    def _get_successor(self, root: Optional['AVLNode[K]']) -> Optional['AVLNode[K]']:
        """Gets the next successor node, ie. the next value greater than root."""
        curr = root.children[1]
        while curr.children[0]:
            curr = curr.children[0]
        return curr

    def _remove(self, root: Optional['AVLNode[K]'], key: K) -> Optional['AVLNode[K]']:
        """Removes the key from root's subtree.
        
        Args: 
            root: The root node of the subtree to remove from.
            key: The desired key to remove.
        """
        if not root: return None
        elif key < root.key or key > root.key:
            index = 0 if key < root.key else 1
            root.children[index] = self._remove(root.children[index], key)

            self._update_height(root)
            root = self._balance(root)
            self._update_height(root)
            return root

        # root is the node to delete
        if not root.children[0] and not root.children[1]:  # root has no children
            return None
        elif root.children[0] and root.children[1]:  # root has both children
            successor = self._get_successor(root)
            root.key = successor.key
            root.children[1] = self._remove(root.children[1], successor.key)            
        else:  # root has only one child
            index = 0 if root.children[0] else 1
            return root.children[index]

        return root
    
    def _insert_node(self, root: Optional['AVLNode[K]'], key: K):
        """Inserts the key into root's subtree.
        
        Args:
            root: The root of the subtree to insert into.
            node: The node to insert. """
        if not root: return AVLNode(key=key)

        index = 0 if key < root.key else 1
        root.children[index] = self._insert_node(root.children[index], key)

        root = self._balance(root)
        self._update_height(root)
        return root
    
    def _balance(self, root: 'AVLNode[K]'):
        """Balances root'ss subtree if necessary to maintain the AVL tree invariant."""
        weight = self._get_weight(root)
        if weight >= -1 and weight <= 1: return root  # balanced

        # if weight positive, left subtree taller, if weight negative, right subtree taller
        left, right = root.children
        if weight < -1:  # imbalance in right subtree
            if self._get_weight(right) > 0:  # right-left imbalance; double rotate right->left
                root.children[1] = self._rotate(right, Direction.RIGHT)
            root = self._rotate(root, Direction.LEFT)
        elif weight > 1:  # imbalance in left subtree
            if self._get_weight(left) < 0:  # left-right imbalance; double rotate left->right
                root.children[0] = self._rotate(left, Direction.LEFT)
            root = self._rotate(root, Direction.RIGHT)
        
        return root
    
    def _rotate(self, node: 'AVLNode[K]', direction: Direction) -> 'AVLNode[K]':
        """Performs a rotation on the given node.
        
        Args:
            node: The node that violates the AVL tree height invariant.
            direction: The direction to rotation, either left or right.
        
        Returns: 
            The AVLNode for node's position.
        """
        direction_val = direction.value
        opposite = abs(1 - direction_val)

        child = node.children[opposite]
        node.children[opposite] = child.children[direction_val] 
        child.children[direction_val] = node

        self._update_height(node)

        return child
    
    def _get_weight(self, node: 'AVLNode[K]') -> int:
        """Calculates the 'weight' between the given node's left and right subtrees.
        AVL invariant mandates that the difference in height between the left and
        right subtrees is no greater than one."""
        left, right = node.children
        left_height = left.height if left else -1
        right_height = right.height if right else -1

        return left_height - right_height
    
    def _update_height(self, node: 'AVLNode[K]'):
        """Updates the given node's height based upon it's children."""
        def _get_child_height(node: 'AVLNode[K]', index: int) -> int:
            return node.children[index].height if node.children[index] else -1
    
        node.height = 1 + max(_get_child_height(node, 0), _get_child_height(node, 1))


class TestAVLTree(unittest.TestCase):
    def test_constructor_empty(self):
        avl = AVLTree()

        self.assertTrue(avl._root is None)
        self.assertTrue(avl.size == 0)
    
    def test_constructor_sequence(self):
        sequence = [5, 2, 8, 6, 9, 7]
        avl = AVLTree(sequence)

        root = AVLNode(key=6, height=2)
        left = AVLNode(key=5, height=1)
        right = AVLNode(key=8, height=1)
        left_left = AVLNode(key=2, height=0)
        right_left = AVLNode(key=7, height=0)
        right_right = AVLNode(key=9, height=0)
        root.children = [left, right]
        left.children = [left_left, None]
        right.children = [right_left, right_right]

        self.assertEqual(avl._root, root)

    def test_insert(self):
        avl = AVLTree()
        avl.insert(1)
        avl.insert(0)
        avl.insert(2)
        
        root = AVLNode(key=1, height=1)
        left = AVLNode(key=0, height=0)
        right = AVLNode(key=2, height=0)
        root.children = [left, right]

        self.assertEqual(avl._root, root)
    
    def test_insert_exists(self):
        avl = AVLTree()
        avl.insert(1)
        avl.insert(1)

        root = AVLNode(key=1, height=0)

        self.assertEqual(avl._root, root)
    
    def test_contains(self):
        avl = AVLTree()
        avl.insert(1)
        avl.insert(0)
        avl.insert(2)

        self.assertTrue(avl.contains(2))
        self.assertFalse(avl.contains(3))

    def test_left_rotation(self):
        avl = AVLTree()
        avl.insert(0)
        avl.insert(1)
        avl.insert(2)

        root = AVLNode(key=1, height=1)
        left = AVLNode(key=0, height=0)
        right = AVLNode(key=2, height=0)
        root.children = [left, right]

        self.assertEqual(avl._root, root)
    
    def test_right_rotation(self):
        avl = AVLTree()
        avl.insert(2)
        avl.insert(1)
        avl.insert(0)

        root = AVLNode(key=1, height=1)
        left = AVLNode(key=0, height=0)
        right = AVLNode(key=2, height=0)
        root.children = [left, right]

        self.assertEqual(avl._root, root)
    
    def test_left_right_rotation(self):
        avl = AVLTree()
        avl.insert(3)
        avl.insert(1)
        avl.insert(2)

        root = AVLNode(key=2, height=1)
        left = AVLNode(key=1, height=0)
        right = AVLNode(key=3, height=0)
        root.children = [left, right]

        self.assertEqual(avl._root, root)
    
    def test_right_left_rotation(self):
        avl = AVLTree()
        avl.insert(1)
        avl.insert(3)
        avl.insert(2)

        root = AVLNode(key=2, height=1)
        left = AVLNode(key=1, height=0)
        right = AVLNode(key=3, height=0)

        root.children = [left, right]

        self.assertEqual(avl._root, root)
    
    def test_left_right_rotation_l(self):
        avl = AVLTree()
        avl.insert(5)
        avl.insert(2)
        avl.insert(8)
        avl.insert(6)
        avl.insert(9)
        avl.insert(7)

        root = AVLNode(key=6, height=2)
        left = AVLNode(key=5, height=1)
        right = AVLNode(key=8, height=1)
        left_left = AVLNode(key=2, height=0)
        right_left = AVLNode(key=7, height=0)
        right_right = AVLNode(key=9, height=0)

        root.children = [left, right]
        left.children = [left_left, None]
        right.children = [right_left, right_right]

        self.assertEqual(avl._root, root)
    
    def test_right_left_rotation_l(self):
        avl = AVLTree()
        avl.insert(6)
        avl.insert(2)
        avl.insert(7)
        avl.insert(1)
        avl.insert(4)
        avl.insert(5)

        root = AVLNode(key=4, height=2)
        left = AVLNode(key=2, height=1)
        right = AVLNode(key=6, height=1)
        left_left = AVLNode(key=1, height=0)
        right_left = AVLNode(key=5, height=0)
        right_right = AVLNode(key=7, height=0)

        root.children = [left, right]
        left.children = [left_left, None]
        right.children = [right_left, right_right]

        self.assertEqual(avl._root, root)
    
    def test_remove_left(self):
        avl = AVLTree()
        avl.insert(1)
        avl.insert(0)
        avl.remove(1)

        root = AVLNode(key=0, height=0)
        self.assertEqual(avl._root, root)
    
    def test_remove_right(self):
        avl = AVLTree()
        avl.insert(0)
        avl.insert(1)
        avl.remove(0)

        root = AVLNode(key=1, height=0)
        self.assertEqual(avl._root, root)

    def test_remove_root(self):
        avl = AVLTree()
        avl.insert(0)
        avl.remove(0)

        self.assertEqual(avl._root, None)
    
    def test_remove_both(self):
        avl = AVLTree()
        avl.insert(1)
        avl.insert(0)
        avl.insert(2)
        avl.remove(1)

        root = AVLNode(key=2, height=1)
        left = AVLNode(key=0, height=0)
        root.children[0] = left

        self.assertEqual(avl._root, root)
    
    def test_remove_dne(self):
        avl = AVLTree()
        avl.insert(2)
        avl.insert(1)
        avl.remove(0)

        root = AVLNode(key=2, height=1)
        left = AVLNode(key=1, height=0)
        root.children = [left, None]

        self.assertEqual(avl._root, root)
    
    def test_remove_rotation(self):
        avl = AVLTree()
        avl.insert(6)
        avl.insert(5)
        avl.insert(8)
        avl.insert(2)
        avl.insert(7)
        avl.insert(9)
        avl.remove(2)
        avl.remove(5)

        root = AVLNode(key=8, height=2)
        left = AVLNode(key=6, height=1)
        right = AVLNode(key=9, height=0)
        left_right = AVLNode(key=7, height=0)
        root.children = [left, right]
        left.children = [None, left_right]

        print(f"AVL: {avl._root}")
        print(f"MOCK: {root}")
        self.assertEqual(avl._root, root)
    
    def test_get_sorted_keys(self):
        avl = AVLTree()
        avl.insert(6)
        avl.insert(5)
        avl.insert(8)
        avl.insert(2)
        avl.insert(7)
        avl.insert(9)
        keys = avl.get_sorted_keys()

        expected = [2, 5, 6, 7, 8, 9]
        self.assertEqual(keys, expected)


if __name__ == '__main__':
    unittest.main()

