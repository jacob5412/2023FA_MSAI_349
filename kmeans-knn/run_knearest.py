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
from k_nearest_neighbor import KNearestNeighbor


def read_data(file_name):
    """
    Read data from a CSV file and return it as a list of label-feature pairs.

    Args:
        file_name (str): The name of the CSV file to read.

    Returns:
        list: A list of label-feature pairs.
    """
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
    """
    Extract numerical labels from a list of label-feature pairs.

    Args:
        data_set (list): A list of label-feature pairs.

    Returns:
        list: A list of numerical labels.
    """
    labels = [int(row[0]) for row in data_set]
    return labels


def get_numerical_features(data_set):
    """
    Extract numerical features from a list of label-feature pairs.

    Args:
        data_set (list): A list of label-feature pairs.

    Returns:
        list: A list of numerical features.
    """
    features = [[int(datapoint) for datapoint in row[1]] for row in data_set]
    return features


if __name__ == "__main__":
    train_set = read_data("valid.csv")
    train_set_labels = np.array(get_numerical_labels(train_set))
    train_set_features = np.array(get_numerical_features(train_set))

    validation_set = read_data("valid.csv")
    validation_set_labels = np.array(get_numerical_labels(validation_set))
    validation_set_features = np.array(get_numerical_features(validation_set))

    
    knearest = KNearestNeighbor(5)
    knearest.fit(train_set_features,train_set_labels)
    predicted_labels = knearest.predict(
        validation_set_features
    )

    num_classes = max(max(validation_set_labels), max(predicted_labels))+1
    confusion_matrix = create_confusion_matrix(num_classes, validation_set_labels, predicted_labels)

    display_confusion_matrix(confusion_matrix)

    eval_metrics = eval_metrics_from_confusion_matrix(confusion_matrix)

    display_eval_metrics(eval_metrics)