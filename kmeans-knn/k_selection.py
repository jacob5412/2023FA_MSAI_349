"""
Get best k for k-means or KNNs
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
logger = logging.getLogger("k-selection")


def get_best_k(
    training_set_features: np.ndarray,
    validation_set_features: np.ndarray,
    validation_set_labels: np.ndarray,
    num_classes: int,
    k_list: list,
    algorithm: [KMeans, KNearestNeighbor],
    pca_num_components: int,
    scaler: str,
) -> int:
    avg_accuracies = {}
    num_iterations = 10
    training_indices = np.arange(training_set_features.shape[0])
    validation_indices = np.arange(validation_set_features.shape[0])

    for k in k_list:
        total_accuracy = 0
        for _ in range(num_iterations):
            # Shuffle Data
            np.random.shuffle(training_indices)
            np.random.shuffle(validation_indices)
            shuffled_training_set_features = training_set_features[training_indices]
            shuffled_validation_set_features = validation_set_features[
                validation_indices
            ]
            shuffled_validation_set_labels = validation_set_labels[validation_indices]

            # Scale Data
            if scaler == "GrayscaleScaler":
                scaler_obj = GrayscaleScaler()
                scaled_training_set_features = scaler_obj.fit_transform(
                    shuffled_training_set_features
                )
                scaled_validation_set_features = scaler_obj.fit_transform(
                    shuffled_validation_set_features
                )
            elif scaler == "StandardScaler":
                scaler_obj = StandardScaler()
                scaler_obj.fit(shuffled_training_set_features)
                scaled_training_set_features = scaler_obj.transform(
                    shuffled_training_set_features
                )
                scaled_validation_set_features = scaler_obj.transform(
                    shuffled_validation_set_features
                )

            # Fit PCA
            pca = PCA(pca_num_components)
            pca.fit(scaled_training_set_features)
            transformed_train_features = pca.transform(scaled_training_set_features)
            transformed_valid_features = pca.transform(scaled_validation_set_features)

            # Train Kmeans
            algorithm_obj = algorithm(k)
            algorithm_obj.fit(transformed_train_features)
            predicted_labels = algorithm_obj.predict(
                transformed_valid_features, shuffled_validation_set_labels
            )

            # Evaluate
            confusion_mat = create_confusion_matrix(
                num_classes, shuffled_validation_set_labels, predicted_labels
            )
            eval_metrics = eval_metrics_from_confusion_matrix(confusion_mat)
            total_accuracy += eval_metrics["overall"]["accuracy"]
        avg_accuracies[k] = total_accuracy / num_iterations

        logger.info(
            "Average Accuracy for k-%d is %.2f",
            k,
            avg_accuracies[k],
        )
    return max(avg_accuracies, key=lambda key: avg_accuracies[key])
