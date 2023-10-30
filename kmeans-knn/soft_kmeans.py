"""
Soft K-means (Probability assignments)
"""
import logging

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("soft_kmeans")


class SoftKMeans:
    """
    This class implements the soft k-means algorithm.

    Args:
        n_clusters (int): The number of clusters to create.
        sharpness (float): A parameter controlling the sharpness of the
                           distribution.
    """

    def __init__(self, n_clusters, sharpness=1.0):
        self.n_clusters = n_clusters
        self.centroids = None
        self.sharpness = sharpness

    def fit(
        self,
        features,
        rtol_threshold=1e-5,
        atol_threshold=1e-7,
        n_iterations=10000,
    ):
        """
        Fit the Soft K-means model to the input data.

        Args:
            features (np.ndarray): array containing inputs of size
            (n_samples, n_features).
            rtol_threshold (float): Relative tolerance for convergence.
            atol_threshold (float): Absolute tolerance for convergence.
            n_iterations (int): The maximum number of iterations.
        """
        self.centroids = self._initialize_centroids(features)
        previous_centroids = np.zeros_like(self.centroids)

        while not self._has_converged(
            previous_centroids, rtol_threshold, atol_threshold, n_iterations
        ):
            n_iterations -= 1
            previous_centroids = self.centroids
            softmax_probabilities = self._update_softmax_probabilities(features)
            self._update_centroids(features, softmax_probabilities)

    def _initialize_centroids(self, features):
        """
        Randomly select n_clusters data points from features as initial means.
        """
        random_indices = np.random.choice(
            features.shape[0], self.n_clusters, replace=False
        )
        return features[random_indices].astype(float)

    def _has_converged(
        self, previous_centroids, rtol_threshold, atol_threshold, n_iterations
    ):
        """
        Soft K-means converges after n_iterations or when current centroid and
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

    def _update_softmax_probabilities(self, features):
        """
        Update the softmax probabilities for each data point.
        """
        # z_im = - beta * || x_i - mu_m ||_2
        z_im = -self.sharpness * np.sum(
            (features[:, np.newaxis] - self.centroids) ** 2, axis=2
        )
        # softmax(z_im) = e^(z_im - max(z_im)) / sum(e^(z_im - max(z_im)))
        # Numerically Stable Softmax: https://jaykmody.com/blog/stable-softmax/
        # https://stackoverflow.com/questions/42599498/numerically-stable-softmax
        softmax_probabilities = np.exp(z_im - z_im.max(axis=1, keepdims=True))
        softmax_probabilities /= softmax_probabilities.sum(axis=1, keepdims=True)
        return softmax_probabilities

    def _update_centroids(self, features, softmax_probabilities):
        """
        Update the cluster centroids based on the given probabilities.
        """
        for cluster in range(self.n_clusters):
            weighted_sum = np.sum(
                softmax_probabilities[:, cluster][:, np.newaxis] * features, axis=0
            )
            total_weight = softmax_probabilities[:, cluster].sum()
            if total_weight > 0:
                self.centroids[cluster] = weighted_sum / total_weight

    def predict(self, features, labels):
        """
        Predict cluster assignments based on majority voting within each
        cluster.

        Args:
            features (np.ndarray): array containing inputs of size
            (n_samples, n_features).
            labels (np.ndarray): array containing input labels.

        Returns:
            predictions (np.ndarray): predicted cluster membership for each
            feature, of size (n_samples,). Each element of the array is the
            index of the cluster the sample belongs to.
        """
        softmax_probabilities = self._update_softmax_probabilities(features)
        cluster_assignments = np.argmax(softmax_probabilities, axis=1)
        predicted_labels = np.zeros_like(cluster_assignments)
        cluster_labels = np.unique(cluster_assignments)

        for cluster_id in cluster_labels:
            cluster_mask = cluster_assignments == cluster_id
            if np.sum(cluster_mask) > 0:
                true_labels = labels[cluster_mask]
                votes = np.bincount(true_labels)
                majority_label = np.argmax(votes)
                predicted_labels[cluster_mask] = majority_label
        return predicted_labels