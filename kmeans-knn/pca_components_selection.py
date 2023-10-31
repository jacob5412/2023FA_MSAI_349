"""
selecting the best number of PCA components based on the average
accuracy over multiple iterations.
"""
import logging

import numpy as np

from k_nearest_neighbor import KNearestNeighbor
from kmeans import KMeans
from utilities.evaluation_utils import (
    create_confusion_matrix,
    eval_metrics_from_confusion_matrix,
)
from utilities.pca_utils import PCA

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pca-components-selection")


def get_best_pca_components(
    training_set_features: np.ndarray,
    validation_set_features: np.ndarray,
    validation_set_labels: np.ndarray,
    num_classes: int,
    n_components_list: list,
    algorithm: [KMeans, KNearestNeighbor],
) -> int:
    """
    Selects the best number of PCA components from a list of possible
    component values based on the average accuracy over multiple iterations.

    Parameters:
        training_set_features (np.ndarray): Features of the training set.
        validation_set_features (np.ndarray): Features of the validation set.
        validation_set_labels (np.ndarray): Labels of the validation set.
        num_classes (int): Number of classes in the classification task.
        n_components_list (list): A list of possible PCA component values to
        consider.
        algorithm (KMeans or KNearestNeighbor): The machine learning algorithm
        used for classification.

    Returns:
        int: The optimal number of PCA components for the given classification
        task.
    """
    best_accuracy = 0
    best_pca_ncomponents = None
    num_iterations = 10
    training_indices = np.arange(training_set_features.shape[0])
    validation_indices = np.arange(validation_set_features.shape[0])

    for n_components in n_components_list:
        total_accuracy = 0
        pca = PCA(n_components)

        for _ in range(num_iterations):
            # Shuffle data
            np.random.shuffle(training_indices)
            np.random.shuffle(validation_indices)
            shuffled_training_set_features = training_set_features[training_indices]
            shuffled_validation_set_features = validation_set_features[
                validation_indices
            ]
            shuffled_validation_set_labels = validation_set_labels[validation_indices]

            # Fit PCA
            pca.fit(shuffled_training_set_features)
            transformed_train_features = pca.transform(shuffled_training_set_features)
            transformed_valid_features = pca.transform(shuffled_validation_set_features)

            # Train Kmeans
            algorithm.fit(transformed_train_features)
            predicted_labels = algorithm.predict(
                transformed_valid_features, shuffled_validation_set_labels
            )

            # Evaluate
            confusion_mat = create_confusion_matrix(
                num_classes, shuffled_validation_set_labels, predicted_labels
            )
            eval_metrics = eval_metrics_from_confusion_matrix(confusion_mat)
            total_accuracy += eval_metrics["overall"]["accuracy"]

        average_accuracy = total_accuracy / num_iterations
        logger.info(
            "Average Accuracy for %d components is %.2f", n_components, average_accuracy
        )
        if average_accuracy > best_accuracy:
            best_accuracy = average_accuracy
            best_pca_ncomponents = n_components

    return best_pca_ncomponents
