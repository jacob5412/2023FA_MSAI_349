"""
K-means
"""
import distance_utils
import numpy as np


class KMeans:
    def __init__(self, n_clusters):
        """
        This class implements the traditional KMeans algorithm with hard
        assignments:

        https://en.wikipedia.org/wiki/K-means_clustering

        The KMeans algorithm has two steps:

        1. Update assignments
        2. Update the means

        While you only have to implement the fit and predict functions to pass
        the test cases, we recommend that you use an update_assignments
        function and an update_means function internally for the class.

        Use only numpy to implement this algorithm.

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
        rtol_threshold=1e-3,
        atol_threshold=1e-4,
        n_iterations=1000,
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

    def predict(self, features, metric="euclidean"):
        """
        Given features, an np.ndarray of size (n_samples, n_features), predict
        cluster membership labels.

        Args:
            features (np.ndarray): array containing inputs of size
            (n_samples, n_features).
        Returns:
            predictions (np.ndarray): predicted cluster membership for each
            features, of size (n_samples,). Each element of the array is the
            index of the cluster the sample belongs to.
        """
        return self._update_cluster_assignments(features, metric)

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
