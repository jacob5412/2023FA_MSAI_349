"""
Run K-means algorithm
"""
import logging

import numpy as np

from k_selection import get_best_k
from kmeans import KMeans
from pca_components_selection import get_best_pca_components
from scaler_selection import get_best_scaler
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
logger = logging.getLogger("kmeans-training")


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

    # Hyperparameter-Tuning
    best_pca_num_components = get_best_pca_components(
        training_set_features,
        validation_set_features,
        validation_set_labels,
        NUM_CLASSES,
        [500, 550, 600, 650, 700, 750],
        KMeans(11),
    )
    logger.info("PCA with %d components performed the best.", best_pca_num_components)
    best_scaler = get_best_scaler(
        training_set_features,
        validation_set_features,
        validation_set_labels,
        NUM_CLASSES,
        KMeans(11),
        best_pca_num_components,
    )
    logger.info("%s performed the best.", best_scaler)
    best_k = get_best_k(
        training_set_features,
        validation_set_features,
        validation_set_labels,
        NUM_CLASSES,
        [10, 11, 12, 13, 14, 15],
        KMeans,
        best_pca_num_components,
        best_scaler,
    )
    logger.info("%d performed the best.", best_k)

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

    pca = PCA(best_pca_num_components)
    pca.fit(scaled_training_set_features)

    transformed_train_features = pca.transform(scaled_training_set_features)
    transformed_test_features = pca.transform(scaled_testing_set_features)

    kmeans = KMeans(best_k)
    kmeans.fit(transformed_train_features)
    predicted_labels = kmeans.predict(transformed_test_features, testing_set_labels)
    confusion_mat = create_confusion_matrix(
        NUM_CLASSES, testing_set_labels, predicted_labels
    )
    display_confusion_matrix(confusion_mat)
    eval_metrics = eval_metrics_from_confusion_matrix(confusion_mat)
    display_eval_metrics(eval_metrics)
