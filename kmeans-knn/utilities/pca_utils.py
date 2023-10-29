"""
Principle Component Analysis
"""
import numpy as np


class PCA:
    """
    Performs PCA to reduce the dimensionality of the input features.
    """

    def __init__(self, num_components):
        self.num_components = num_components

    def fit(self, features):
        """
        Fit the PCA model to the input data and reduce its dimensionality.

        Parameters:
            features (np.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Reduced feature data of shape
                        (n_samples, num_components).
        """
        # Center the data by subtracting the mean from each feature.
        features_meaned = features - np.mean(features, axis=0)

        # Calculate the covariance matrix of the mean-centered data.
        cov_mat = np.cov(features_meaned, rowvar=False)

        # Calculate Eigenvalues and Eigenvectors of the covariance matrix
        eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)

        # Sort the Eigenvectors in descending order
        sorted_indices = np.argsort(eigen_values)[::-1]
        eigen_vectors_sorted = eigen_vectors[:, sorted_indices]

        # Select the first n Eigenvectors
        eigen_vectors_subset = eigen_vectors_sorted[:, : self.num_components]

        # Transfrom the data
        features_reduced = np.dot(features_meaned, eigen_vectors_subset)
        return features_reduced
