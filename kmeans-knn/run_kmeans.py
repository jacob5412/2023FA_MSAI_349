"""
Run K-means algorithm
"""
import logging

import numpy as np
from kmeans import KMeans
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


def get_best_scaler(
    training_set_features, validation_set_features, validation_set_labels
):
    n_clusters = 10

    # Scaling
    standard_scaler = StandardScaler()
    standard_scaler.fit(training_set_features)
    scaled_training_set_features = standard_scaler.transform(training_set_features)
    scaled_validation_set_features = standard_scaler.transform(validation_set_features)

    # Reducing Dimensions
    std_pca = PCA(num_components=600)
    std_pca.fit(scaled_training_set_features)
    std_pca_training_set_features = std_pca.transform(scaled_training_set_features)
    std_pca_validation_set_features = std_pca.transform(scaled_validation_set_features)

    # Fitting data
    kmeans = KMeans(n_clusters)
    kmeans.fit(std_pca_training_set_features)
    predicted_labels = kmeans.predict(
        std_pca_validation_set_features, validation_set_labels
    )
    confusion_mat = create_confusion_matrix(
        n_clusters, validation_set_labels, predicted_labels
    )
    eval_metrics = eval_metrics_from_confusion_matrix(confusion_mat)
    std_pca_accuracy = eval_metrics["overall"]["accuracy"]
    logger.info("Standard Scaler overall accuracy: %s", std_pca_accuracy)

    # Grayscaled Dimensions
    grayscale_scaler = GrayscaleScaler()
    scaled_training_set_features = grayscale_scaler.fit_transform(training_set_features)
    scaled_validation_set_features = grayscale_scaler.fit_transform(
        validation_set_features
    )

    # Reducing Dimensions
    gtd_pca = PCA(num_components=600)
    gtd_pca.fit(scaled_training_set_features)
    gtd_pca_training_set_features = gtd_pca.transform(scaled_training_set_features)
    gtd_pca_validation_set_features = gtd_pca.transform(scaled_validation_set_features)

    # Fitting data
    kmeans = KMeans(n_clusters)
    kmeans.fit(gtd_pca_training_set_features)
    predicted_labels = kmeans.predict(
        gtd_pca_validation_set_features, validation_set_labels
    )
    confusion_mat = create_confusion_matrix(
        n_clusters, validation_set_labels, predicted_labels
    )
    eval_metrics = eval_metrics_from_confusion_matrix(confusion_mat)
    gtd_pca_accuracy = eval_metrics["overall"]["accuracy"]
    logger.info("Grayscaled Scaler overall accuracy: %s", gtd_pca_accuracy)

    return (
        (standard_scaler, std_pca, "standard_scaler")
        if std_pca_accuracy >= gtd_pca_accuracy
        else (grayscale_scaler, gtd_pca, "grayscaled_scaler")
    )


if __name__ == "__main__":
    training_set = read_data("mnist_dataset/train.csv")
    training_set_labels = np.array(get_numerical_labels(training_set))
    training_set_features = np.array(get_numerical_features(training_set))
    validation_set = read_data("mnist_dataset/valid.csv")
    validation_set_labels = np.array(get_numerical_labels(validation_set))
    validation_set_features = np.array(get_numerical_features(validation_set))
    (trained_scaler, trained_pca, scaler_type) = get_best_scaler(
        training_set_features, validation_set_features, validation_set_labels
    )
    print(f"Selecting {scaler_type} for future operations.")
