"""
Run K-means algorithm
"""
import numpy as np
from evaluation_utils import (
    create_confusion_matrix,
    display_confusion_matrix,
    display_eval_metrics,
    eval_metrics_from_confusion_matrix,
)
from kmeans import KMeans


def read_data(file_name):
    data_set = []
    with open(file_name, "rt") as f:
        for line in f:
            line = line.replace("\n", "")
            tokens = line.split(",")
            label = tokens[0]
            attribs = []
            for i in range(784):
                attribs.append(tokens[i + 1])
            data_set.append([label, attribs])
    return data_set


def get_numerical_labels(data_set):
    labels = [int(row[0]) for row in data_set]
    return labels


def get_numerical_features(data_set):
    features = [[int(datapoint) for datapoint in row[1]] for row in data_set]
    return features


if __name__ == "__main__":
    validation_set = read_data("valid.csv")
    validation_set_labels = np.array(get_numerical_labels(validation_set))
    validation_set_features = np.array(get_numerical_features(validation_set))

    n_clusters = 10
    kmeans = KMeans(n_clusters)
    kmeans.fit(validation_set_features, metric="euclidean")
    predicted_labels = kmeans.predict(validation_set_features, metric="euclidean")

    confusion_mat = create_confusion_matrix(
        n_clusters, validation_set_labels, predicted_labels
    )
    display_confusion_matrix(confusion_mat)
    eval_metrics = eval_metrics_from_confusion_matrix(confusion_mat)
    display_eval_metrics(eval_metrics)
