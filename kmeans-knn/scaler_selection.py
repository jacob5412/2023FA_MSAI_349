"""
Selecting the best scaler (StandardScaler or GrayscaleScaler) based on
the average accuracy over multiple iterations.
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
from utilities.preprocessing_utils import GrayscaleScaler, StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("scaler-selection")


def get_best_scaler(
    training_set_features: np.ndarray,
    validation_set_features: np.ndarray,
    validation_set_labels: np.ndarray,
    num_classes: int,
    algorithm: [KMeans, KNearestNeighbor],
    pca_num_components: int,
) -> str:
    """
    Selects the best scaler (StandardScaler or GrayscaleScaler) based on the
    average accuracy over multiple iterations.

    Args:
        training_set_features (np.ndarray): Features of the training set.
        validation_set_features (np.ndarray): Features of the validation set.
        validation_set_labels (np.ndarray): Labels of the validation set.
        num_classes (int): Number of classes in the classification task.
        algorithm (KMeans or KNearestNeighbor): The machine learning algorithm
        used for classification.
        pca_num_components (int): The number of components for PCA.

    Returns:
        str: The name of the selected scaler ('StandardScaler' or
        'GrayscaleScaler').
    """
    avg_accuracies = {}
    num_iterations = 10
    training_indices = np.arange(training_set_features.shape[0])
    validation_indices = np.arange(validation_set_features.shape[0])

    pca = PCA(pca_num_components)
    total_accuracy = 0
    for _ in range(num_iterations):
        # Shuffle Data
        np.random.shuffle(training_indices)
        np.random.shuffle(validation_indices)
        shuffled_training_set_features = training_set_features[training_indices]
        shuffled_validation_set_features = validation_set_features[validation_indices]
        shuffled_validation_set_labels = validation_set_labels[validation_indices]

        # Scale Data
        standard_scaler = StandardScaler()
        standard_scaler.fit(shuffled_training_set_features)
        scaled_training_set_features = standard_scaler.transform(
            shuffled_training_set_features
        )
        scaled_validation_set_features = standard_scaler.transform(
            shuffled_validation_set_features
        )

        # Fit PCA
        pca.fit(scaled_training_set_features)
        transformed_train_features = pca.transform(scaled_training_set_features)
        transformed_valid_features = pca.transform(scaled_validation_set_features)

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
    avg_accuracies["StandardScaler"] = total_accuracy / num_iterations

    logger.info(
        "Average Accuracy for StandardScaler is %.2f", avg_accuracies["StandardScaler"]
    )

    pca = PCA(pca_num_components)
    total_accuracy = 0
    for _ in range(num_iterations):
        # Shuffle Data
        np.random.shuffle(training_indices)
        np.random.shuffle(validation_indices)
        shuffled_training_set_features = training_set_features[training_indices]
        shuffled_validation_set_features = validation_set_features[validation_indices]
        shuffled_validation_set_labels = validation_set_labels[validation_indices]

        # Scale Data
        grayscale_scaler = GrayscaleScaler()
        scaled_training_set_features = grayscale_scaler.fit_transform(
            shuffled_training_set_features
        )
        scaled_validation_set_features = grayscale_scaler.fit_transform(
            shuffled_validation_set_features
        )

        # Fit PCA
        pca.fit(scaled_training_set_features)
        transformed_train_features = pca.transform(scaled_training_set_features)
        transformed_valid_features = pca.transform(scaled_validation_set_features)

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
    avg_accuracies["GrayscaleScaler"] = total_accuracy / num_iterations

    logger.info(
        "Average Accuracy for GrayscaleScaler is %.2f",
        avg_accuracies["GrayscaleScaler"],
    )

    return max(avg_accuracies, key=lambda key: avg_accuracies[key])
