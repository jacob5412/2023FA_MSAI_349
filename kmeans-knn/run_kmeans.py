import numpy as np

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
    validation_set_labels = np.array(get_labels(validation_set))
    validation_set_features = np.array(get_features(validation_set))
