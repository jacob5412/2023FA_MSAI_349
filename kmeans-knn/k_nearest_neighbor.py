"""
K-Nearest Neighbor Classifer
"""
import numpy as np

from collections import Counter
from utilities import distance_utils


class KNearestNeighbor:
    """
    A K-Nearest Neighbor Classifier.

    Attributes:
        n_neighbors (int): The number of nearest neighbors to consider during
        prediction.
        aggregator (str): The method for aggregating neighbor labels.
        metric (str): The distance metric used for comparing data points.

    """

    def __init__(self, n_neighbors, aggregator="mode", metric="euclidean"):
        self.n_neighbors = n_neighbors
        self.aggregator = aggregator
        self.metric = metric

    def _get_distances(self, train_feature, test_feature, metric):
        """
        Calculate the distances between training and test features.

        Args:
            train_feature (numpy.ndarray): The feature from the training set.
            test_feature (numpy.ndarray): The feature from the test set.
            metric (str): The distance metric to use for the calculation.

        Returns:
            numpy.ndarray: An array of distances between the two features.

        """
        distance_metric = getattr(distance_utils, metric + "_distance")
        distances = distance_metric(train_feature, test_feature)
        return distances

    def _label_voting(self, neighbors):
        """
        Perform label voting among the nearest neighbors.

        Args:
            neighbors (numpy.ndarray): An array of labels from the
            nearest neighbors.

        Returns:
            The label with the highest vote.
        """
        if self.aggregator == "mode":
            label_counts = Counter(neighbors)
            most_common_labels = label_counts.most_common()
            return most_common_labels[0][0]

    def fit(
        self,
        features,
        labels,
    ):
        """
        Fit the K-Nearest Neighbor classifier with training data.

        Args:
            features (numpy.ndarray): The features from the training set.
            labels (list or numpy.ndarray): The corresponding labels for the
            training data.

        """
        self.features = features
        self.labels = labels

    def predict(self, query, ignore_first=False, metric="euclidean"):
        """
        Predict labels for a query or a batch of queries.

        Args:
            query (numpy.ndarray): The query for which labels are to be
            predicted.
            ignore_first (bool): Whether to ignore the first neighbor label
            (False by default).
            metric (str): The distance metric to use for the prediction.

        Returns:
            The predicted labels for the query
        """
        distances = self._get_distances(self.features, query, metric)
        sorted_indices = np.argsort(distances, axis=0)
        sorted_labels = np.take(np.array(self.labels), sorted_indices, axis=0)
        neighbors = sorted_labels[: self.n_neighbors]
        predicted_labels = np.apply_along_axis(
            self._label_voting,
            axis=0,
            arr=neighbors[1:] if ignore_first else neighbors,
        )
        return predicted_labels
