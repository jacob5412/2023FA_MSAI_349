"""
Run K-means algorithm
"""
import logging

import numpy as np
from k_nearest_neighbor import KNearestNeighbor
from knns_hyperparams import (
    get_best_distance,
    get_best_k,
    get_best_pca_components,
    get_best_scaler,
)
from utilities.evaluation_utils import (
    create_confusion_matrix,
    display_confusion_matrix,
    display_eval_metrics,
    eval_metrics_from_confusion_matrix,
)
from utilities.pca_utils import PCA
from utilities.preprocessing_utils import GrayscaleScaler, StandardScaler
from utilities.read_data import get_numerical_features, get_numerical_labels, read_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("knns-training")


if __name__ == "__main__":
    training_set = read_data("mnist_dataset/train.csv")
    training_set_labels = np.array(get_numerical_labels(training_set))
    training_set_features = np.array(get_numerical_features(training_set))
    validation_set = read_data("mnist_dataset/valid.csv")
    validation_set_labels = np.array(get_numerical_labels(validation_set))
    validation_set_features = np.array(get_numerical_features(validation_set))
    testing_set = read_data("mnist_dataset/test.csv")
    testing_set_labels = np.array(get_numerical_labels(testing_set))
    testing_set_features = np.array(get_numerical_features(testing_set))

    NUM_CLASSES = 10
    K_COMPONENTS = 11

    # Hyperparameter-Tuning
    best_pca_num_components = get_best_pca_components(
        training_set_features,
        training_set_labels,
        validation_set_features,
        validation_set_labels,
        NUM_CLASSES,
        [None, 500, 550, 600, 650, 700, 750],
        K_COMPONENTS,
    )
    best_pca_num_components = 700  # based on empirical evidence
    logger.info("PCA with %s components performed the best.", best_pca_num_components)
    best_scaler = get_best_scaler(
        training_set_features,
        training_set_labels,
        validation_set_features,
        validation_set_labels,
        NUM_CLASSES,
        K_COMPONENTS,
        best_pca_num_components,  # passing best param
    )
    best_scaler = "GrayscaleScaler"  # based on empirical evidence
    logger.info("%s performed the best.", best_scaler)
    best_k = get_best_k(
        training_set_features,
        training_set_labels,
        validation_set_features,
        validation_set_labels,
        NUM_CLASSES,
        [10, 11, 12, 13, 14, 15],
        best_pca_num_components,  # passing best param
        best_scaler,  # passing best param
    )
    best_k = 14  # based on empirical evidence
    logger.info("%d performed the best.", best_k)
    best_distance_metric = get_best_distance(
        training_set_features,
        training_set_labels,
        validation_set_features,
        validation_set_labels,
        NUM_CLASSES,
        best_k,  # passing best param
        best_pca_num_components,  # passing best param
        best_scaler,  # passing best param
    )
    best_distance_metric = "cosine"  # based on empirical evidence
    logger.info("%s performed the best.", best_distance_metric)

    # Training & testing final K-means
    if best_scaler == "GrayscaleScaler":
        grayscale_scaler = GrayscaleScaler()
        scaled_training_set_features = grayscale_scaler.fit_transform(
            training_set_features
        )
        scaled_testing_set_features = grayscale_scaler.fit_transform(
            testing_set_features
        )
    elif best_scaler == "StandardScaler":
        standard_scaler = StandardScaler()
        standard_scaler.fit(training_set_features)
        scaled_training_set_features = standard_scaler.transform(training_set_features)
        scaled_testing_set_features = standard_scaler.transform(testing_set_features)

    if best_pca_num_components is None:
        transformed_train_features = scaled_training_set_features
        transformed_test_features = scaled_testing_set_features
    else:
        pca = PCA(best_pca_num_components)
        pca.fit(scaled_training_set_features)
        transformed_train_features = pca.transform(scaled_training_set_features)
        transformed_test_features = pca.transform(scaled_testing_set_features)

    knns = KNearestNeighbor(best_k)
    knns.fit(transformed_train_features, training_set_labels)
    predicted_labels = knns.predict(
        transformed_test_features, best_distance_metric
    )
    confusion_mat = create_confusion_matrix(
        NUM_CLASSES, testing_set_labels, predicted_labels
    )
    display_confusion_matrix(confusion_mat)
    eval_metrics = eval_metrics_from_confusion_matrix(confusion_mat)
    display_eval_metrics(eval_metrics)
