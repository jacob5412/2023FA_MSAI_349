"""
Selecting the best hyperparameters for Soft K-means based on
the average accuracy over multiple iterations.
"""
import logging

import numpy as np
from soft_kmeans import SoftKMeans
from utilities.evaluation_utils import (
    create_confusion_matrix,
    eval_metrics_from_confusion_matrix,
)
from utilities.pca_utils import PCA
from utilities.preprocessing_utils import GrayscaleScaler, StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hyperparameter-selection")


def get_best_scaler(
    training_set_features: np.ndarray,
    validation_set_features: np.ndarray,
    validation_set_labels: np.ndarray,
    num_classes: int,
    k_components: int,
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
        k_components: number of k for K-means.
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
        if pca_num_components is None:
            transformed_train_features = shuffled_training_set_features
            transformed_valid_features = shuffled_validation_set_features
        else:
            pca.fit(scaled_training_set_features)
            transformed_train_features = pca.transform(scaled_training_set_features)
            transformed_valid_features = pca.transform(scaled_validation_set_features)

        # Train SoftKMeans
        soft_kmeans = SoftKMeans(k_components)
        soft_kmeans.fit(transformed_train_features)
        predicted_labels = soft_kmeans.predict(
            transformed_valid_features, shuffled_validation_set_labels
        )

        # Evaluate
        confusion_mat = create_confusion_matrix(
            num_classes, shuffled_validation_set_labels, predicted_labels
        )
        eval_metrics = eval_metrics_from_confusion_matrix(confusion_mat)
        total_accuracy += eval_metrics["overall"]["accuracy"]
    avg_accuracies["StandardScaler"] = total_accuracy / num_iterations

    logger.info("Accuracy for StandardScaler is %.3f", avg_accuracies["StandardScaler"])

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
        if pca_num_components is None:
            transformed_train_features = shuffled_training_set_features
            transformed_valid_features = shuffled_validation_set_features
        else:
            pca.fit(scaled_training_set_features)
            transformed_train_features = pca.transform(scaled_training_set_features)
            transformed_valid_features = pca.transform(scaled_validation_set_features)

        # Train SoftKMeans
        soft_kmeans = SoftKMeans(k_components)
        soft_kmeans.fit(transformed_train_features)
        predicted_labels = soft_kmeans.predict(
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
        "Accuracy for GrayscaleScaler is %.3f",
        avg_accuracies["GrayscaleScaler"],
    )

    return max(avg_accuracies, key=lambda key: avg_accuracies[key])


def get_best_k(
    training_set_features: np.ndarray,
    validation_set_features: np.ndarray,
    validation_set_labels: np.ndarray,
    num_classes: int,
    k_list: list,
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
            if pca_num_components is None:
                transformed_train_features = shuffled_training_set_features
                transformed_valid_features = shuffled_validation_set_features
            else:
                pca = PCA(pca_num_components)
                pca.fit(scaled_training_set_features)
                transformed_train_features = pca.transform(scaled_training_set_features)
                transformed_valid_features = pca.transform(
                    scaled_validation_set_features
                )

            # Train SoftKMeans
            soft_kmeans = SoftKMeans(k)
            soft_kmeans.fit(transformed_train_features)
            predicted_labels = soft_kmeans.predict(
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
            "Accuracy for k-%d is %.3f",
            k,
            avg_accuracies[k],
        )
    return max(avg_accuracies, key=lambda key: avg_accuracies[key])


def get_best_pca_components(
    training_set_features: np.ndarray,
    validation_set_features: np.ndarray,
    validation_set_labels: np.ndarray,
    num_classes: int,
    n_components_list: list,
    k_components: int,
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
        k_components: number of k for K-means.

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
            if n_components is None:
                transformed_train_features = shuffled_training_set_features
                transformed_valid_features = shuffled_validation_set_features
            else:
                pca.fit(shuffled_training_set_features)
                transformed_train_features = pca.transform(
                    shuffled_training_set_features
                )
                transformed_valid_features = pca.transform(
                    shuffled_validation_set_features
                )

            # Train SoftKMeans
            soft_kmeans = SoftKMeans(k_components)
            soft_kmeans.fit(transformed_train_features)
            predicted_labels = soft_kmeans.predict(
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
            "Accuracy for %s components is %.3f", n_components, average_accuracy
        )
        if average_accuracy > best_accuracy:
            best_accuracy = average_accuracy
            best_pca_ncomponents = n_components

    return best_pca_ncomponents
