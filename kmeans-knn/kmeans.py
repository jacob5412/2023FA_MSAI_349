"""
K-means
"""
import logging

import numpy as np

from utilities import distance_utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kmeans")


class KMeans:
    """
    This class implements the traditional KMeans algorithm with hard
    assignments.
    """

    def __init__(self, n_clusters):
        """
        The KMeans algorithm has two steps:

        1. Update assignments
        2. Update the means

        Args:
            n_clusters (int): Number of clusters to cluster the given data
                              into.

        """
        self.n_clusters = n_clusters
        self.centroids = None

    def fit(
        self,
        features,
        metric="euclidean",
        rtol_threshold=1e-5,
        atol_threshold=1e-7,
        n_iterations=10000,
    ):
        """
        Fit KMeans to the given data using `self.n_clusters` number of
        clusters. Features can have greater than 2 dimensions.

        Args:
            features (np.ndarray): array containing inputs of size
            (n_samples, n_features).
        Returns:
            None (saves model - means - internally)
        """

        self.centroids = self._initialize_centroids(features)
        previous_centroids = np.zeros_like(self.centroids)

        while not self._has_converged(
            previous_centroids, rtol_threshold, atol_threshold, n_iterations
        ):
            n_iterations -= 1
            previous_centroids = self.centroids
            cluster_assignments = self._update_cluster_assignments(features, metric)
            self._update_centroids(features, cluster_assignments)

    def predict(self, features, labels, metric="euclidean"):
        """
        Given features, an np.ndarray of size (n_samples, n_features), predict
        labels based on weighted voting.

        Args:
            features (np.ndarray): array containing inputs of size
            (n_samples, n_features).
            labels (np.ndarray): array containing input labels.
            metric (str): distance metric to use (e.g., "euclidean").

        Returns:
            predictions (np.ndarray): predicted cluster membership for each
            feature, of size (n_samples,). Each element of the array is the
            index of the cluster the sample belongs to.
        """
        distance_metric = getattr(distance_utils, metric + "_distance")
        distances = distance_metric(features, self.centroids)
        cluster_assignments = np.argmin(distances, axis=1)
        predicted_labels = np.zeros_like(cluster_assignments)
        cluster_labels = np.unique(cluster_assignments)

        for cluster_id in cluster_labels:
            cluster_mask = cluster_assignments == cluster_id
            if np.sum(cluster_mask) > 0:
                true_labels = labels[cluster_mask]
                weights = 10.0 / (1.0 + distances[cluster_mask, cluster_id])
                weighted_votes = np.bincount(true_labels, weights=weights)
                majority_label = np.argmax(weighted_votes)
                predicted_labels[cluster_mask] = majority_label
        return predicted_labels

    def _initialize_centroids(self, features):
        """
        Randomly select n_clusters data points from features as initial means.
        """
        random_indices = np.random.choice(
            features.shape[0], self.n_clusters, replace=False
        )
        return features[random_indices]

    def _has_converged(
        self, previous_centroids, rtol_threshold, atol_threshold, n_iterations
    ):
        """
        K-means converges after n_iterations or when current centroid and
        previous centroid are within a certain threshold.
        """
        no_centroids_change = np.allclose(
            self.centroids, previous_centroids, rtol=rtol_threshold, atol=atol_threshold
        )
        max_iterations = n_iterations <= 0
        if max_iterations:
            logger.info("Max iterations has reached.")
        if no_centroids_change:
            logger.info("No more change in centroids.")
        return no_centroids_change or max_iterations

    def _update_cluster_assignments(self, features, metric):
        """
        Update cluster assignments based on a distance metric.
        """
        distance_metric = getattr(distance_utils, metric + "_distance")
        distances = distance_metric(features, self.centroids)
        cluster_assignments = np.argmin(distances, axis=1)
        return cluster_assignments

    def _update_centroids(self, features, cluster_assignments):
        """
        Update centroids based on the new cluster assignments.
        """
        self.centroids = np.array(
            [
                np.mean(features[cluster_assignments == i], axis=0)
                for i in range(self.n_clusters)
            ]
        )
