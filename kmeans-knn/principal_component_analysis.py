import numpy as np

"""
Ref: https://www.askpython.com/python/examples/principal-component-analysis
"""


class PCA:
    def __init__(self, X, num_components=5):
        self.X = X
        self.num_components = num_components

    def transform(self):
        # mean Centering the data, 20 examples and 5 variables for each example
        X_meaned = self.X - np.mean(self.X, axis=0)

        # calculating the covariance matrix of the mean-centered data.
        cov_mat = np.cov(X_meaned, rowvar=False)

        # Calculating Eigenvalues and Eigenvectors of the covariance matrix
        eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)

        # sort the eigenvalues in descending order
        sorted_index = np.argsort(eigen_values)[::-1]
        sorted_eigenvalue = eigen_values[sorted_index]
        # similarly sort the eigenvectors
        sorted_eigenvectors = eigen_vectors[:, sorted_index]

        # select the first n eigenvectors, n is desired dimension
        # of our final reduced data.
        eigenvector_subset = sorted_eigenvectors[:, 0 : self.num_components]

        # Transform the data
        x_reduced = np.dot(
            eigenvector_subset.transpose(), X_meaned.transpose()
        ).transpose()

        return x_reduced
