"""
Run KNN algorithm
"""
import numpy as np
from utilities.evaluation_utils import (
    create_confusion_matrix,
    display_confusion_matrix,
    display_eval_metrics,
    eval_metrics_from_confusion_matrix,
)
from k_nearest_neighbor import KNearestNeighbor
from utilities.read_data import get_numerical_features, get_numerical_labels, read_data


if __name__ == "__main__":
    train_set = read_data("valid.csv")
    train_set_labels = np.array(get_numerical_labels(train_set))
    train_set_features = np.array(get_numerical_features(train_set))

    validation_set = read_data("valid.csv")
    validation_set_labels = np.array(get_numerical_labels(validation_set))
    validation_set_features = np.array(get_numerical_features(validation_set))

    knearest = KNearestNeighbor(5)
    knearest.fit(train_set_features, train_set_labels)
    predicted_labels = knearest.predict(validation_set_features)

    confusion_matrix = create_confusion_matrix(
        10, validation_set_labels, predicted_labels
    )
    display_confusion_matrix(confusion_matrix)
    eval_metrics = eval_metrics_from_confusion_matrix(confusion_matrix)
    display_eval_metrics(eval_metrics)
