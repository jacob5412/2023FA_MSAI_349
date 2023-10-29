"""
Run K-means algorithm
"""
import numpy as np
from kmeans import KMeans
from utilities.evaluation_utils import (
    create_confusion_matrix,
    display_confusion_matrix,
    display_eval_metrics,
    eval_metrics_from_confusion_matrix,
)
from utilities.read_data import get_numerical_features, get_numerical_labels, read_data

if __name__ == "__main__":
    validation_set = read_data("valid.csv")
    validation_set_labels = np.array(get_numerical_labels(validation_set))
    validation_set_features = np.array(get_numerical_features(validation_set))

    n_clusters = 10
    kmeans = KMeans(n_clusters)
    kmeans.fit(validation_set_features, metric="euclidean")
    predicted_labels = kmeans.predict(
        validation_set_features, validation_set_labels, metric="euclidean"
    )

    confusion_mat = create_confusion_matrix(
        n_clusters, validation_set_labels, predicted_labels
    )
    display_confusion_matrix(confusion_mat)
    eval_metrics = eval_metrics_from_confusion_matrix(confusion_mat)
    display_eval_metrics(eval_metrics)
