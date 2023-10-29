import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.means = None

    def fit(self, features, max_iters=10):
        np.random.seed(0)
        random_indices = np.random.choice(features.shape[0], self.n_clusters, replace=False)
        self.means = features[random_indices]

        for _ in range(max_iters):
            # Update assignments
            distances = np.linalg.norm(features[:, np.newaxis] - self.means, axis=2)
            labels = np.argmin(distances, axis=1)

            # Update means
            new_means = []
            for i in range(self.n_clusters):
                if np.sum(labels == i) > 0:
                    new_cluster_mean = features[labels == i].mean(axis=0)
                else:
                    # If the cluster is empty, select a random point as the new center
                    new_cluster_mean = features[np.random.choice(features.shape[0])]
                new_means.append(new_cluster_mean)

            new_means = np.array(new_means)

            if np.all(new_means == self.means):
                break

            self.means = new_means

    def predict(self, features):
        distances = np.linalg.norm(features[:, np.newaxis] - self.means, axis=2)
        labels = np.argmin(distances, axis=1)
        return labels

# Load the data from CSV files
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
valid_data = pd.read_csv('valid.csv')

# Extract features from the data
train_features = train_data.values
test_features = test_data.values
valid_features = valid_data.values

# Define the number of clusters
n_clusters = 3  # You can change this to your desired number of clusters

# Initialize and fit the KMeans model
kmeans = KMeans(n_clusters)
kmeans.fit(train_features, max_iters=10)  # Adjust the max_iters value as needed

# Predict clusters for test and validation data
test_predictions = kmeans.predict(test_features)
valid_predictions = kmeans.predict(valid_features)

# Plot the clusters for test and validation data
plt.figure(figsize=(12, 6))

# Test Data
plt.subplot(1, 2, 1)
plt.scatter(test_features[:, 0], test_features[:, 1], c=test_predictions, cmap='viridis')
plt.scatter(kmeans.means[:, 0], kmeans.means[:, 1], c='red', marker='x', s=100)
plt.title('Test Data Clusters')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Validation Data
plt.subplot(1, 2, 2)
plt.scatter(valid_features[:, 0], valid_features[:, 1], c=valid_predictions, cmap='viridis')
plt.scatter(kmeans.means[:, 0], kmeans.means[:, 1], c='red', marker='x', s=100)
plt.title('Validation Data Clusters')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()
