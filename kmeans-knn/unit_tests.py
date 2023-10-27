"""
Unit tests
"""
import unittest
from random import randint

import numpy as np
from sklearn.metrics import pairwise

from distance_utils import cosine_distance, euclidean_distance


class TestDistanceMetrics(unittest.TestCase):
    """
    Unit tests for custom distance metrics (cosine and Euclidean distances)
    and comparisons with scikit-learn's pairwise distances.
    """

    def setUp(self) -> None:
        self.mat1 = np.array([[1, 2, 32], [4, 5, 6]])
        self.mat2 = np.array([[4, 543, 6], [7.2, 8, 9]])
        self.mat3 = np.array([[1.3, 2.4311, 4134], [-341, 5.0, 6.0]])
        random_mat_nrows = randint(1, 10)
        random_mat_ncols = randint(1, 10)
        self.mat4 = np.random.rand(random_mat_nrows, random_mat_ncols)
        self.mat5 = np.random.rand(random_mat_nrows, random_mat_ncols)
        self.mat6 = np.array([[-1.0, 2.4, 4.0, 514.31]])

    def test_euclidean_1(self):
        """
        Test Euclidean distance between mat1 and mat2.
        """
        custom_result = euclidean_distance(self.mat1, self.mat2)
        sklearn_result = pairwise.euclidean_distances(self.mat1, self.mat2)
        self.assertTrue(np.allclose(custom_result, sklearn_result))

    def test_euclidean_2(self):
        """
        Test Euclidean distance between mat2 and mat3.
        """
        custom_result = euclidean_distance(self.mat2, self.mat3)
        sklearn_result = pairwise.euclidean_distances(self.mat2, self.mat3)
        self.assertTrue(np.allclose(custom_result, sklearn_result))

    def test_euclidean_3(self):
        """
        Test Euclidean distance with matrices of different length.
        """
        with self.assertRaises(ValueError):
            euclidean_distance(self.mat3, self.mat6)

    def test_euclidean_4(self):
        """
        Test Euclidean distance with random length matrices.
        """
        custom_result = euclidean_distance(self.mat4, self.mat5)
        sklearn_result = pairwise.euclidean_distances(self.mat4, self.mat5)
        self.assertTrue(np.allclose(custom_result, sklearn_result))

    def test_cosine_distance_1(self):
        """
        Test Cosine distance between mat1 and mat2.
        """
        custom_result = cosine_distance(self.mat1, self.mat2)
        sklearn_result = pairwise.cosine_distances(self.mat1, self.mat2)
        self.assertTrue(np.allclose(custom_result, sklearn_result))

    def test_cosine_distance_2(self):
        """
        Test Cosine distance between mat2 and mat3.
        """
        custom_result = cosine_distance(self.mat2, self.mat3)
        sklearn_result = pairwise.cosine_distances(self.mat2, self.mat3)
        self.assertTrue(np.allclose(custom_result, sklearn_result))

    def test_cosine_distance_3(self):
        """
        Test Cosine distance with matrices of different length.
        """
        with self.assertRaises(ValueError):
            cosine_distance(self.mat3, self.mat6)

    def test_cosine_distance_4(self):
        """
        Test Cosine distance with random length matrices.
        """
        custom_result = cosine_distance(self.mat4, self.mat5)
        sklearn_result = pairwise.cosine_distances(self.mat4, self.mat5)
        self.assertTrue(np.allclose(custom_result, sklearn_result))


if __name__ == "__main__":
    unittest.main()
