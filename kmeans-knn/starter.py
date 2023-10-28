import pandas as pd

import distance_utils as du
import k_nearest_neighbor as k_n_n
import principal_component_analysis as pca
import numpy as np


# returns Euclidean distance between vectors a dn b
def euclidean(a, b):
    return du.euclidean_distance(a, b)


# returns Cosine Similarity between vectors a dn b
def cosim(a, b):
    return du.cosine_distance(a, b)


# returns a list of labels for the query dataset based upon labeled observations in the train dataset.
# metric is a string specifying either "euclidean" or "cosim".
# All hyper-parameters should be hard-coded in the algorithm.
def knn(train, query, metric):
    # hyper-parameters
    num_neighbors = len(train) // 2
    num_components = len(train) // 4

    neighbors = get_neighbors(train, query, num_neighbors)
    k_nearest_neighbor = k_n_n.KNearestNeighbor(
        neighbors, distance_measure=metric, aggregator="mean"
    )
    df = pd.DataFrame()
    for train_row in train:
        pd.concat([df, pd.DataFrame(eval(i) for i in train_row[1])])
    features = pca.PCA(df.to_numpy(), num_components).transform()
    k_nearest_neighbor.fit(features, query)

    return k_nearest_neighbor.predict(features, ignore_first=True)


# returns a list of labels for the query dataset based upon observations in the train dataset.
# labels should be ignored in the training set
# metric is a string specifying either "euclidean" or "cosim".
# All hyper-parameters should be hard-coded in the algorithm.
def kmeans(train, query, metric):
    return labels


# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        train_row_res = np.array([[eval(i) for i in train_row[1]]])
        test_row_res = np.array([[eval(i) for i in test_row[1]]])
        dist = du.euclidean_distance(train_row_res, test_row_res)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors


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


def show(file_name, mode):
    data_set = read_data(file_name)
    for obs in range(len(data_set)):
        for idx in range(784):
            if mode == "pixels":
                if data_set[obs][1][idx] == "0":
                    print(" ", end="")
                else:
                    print("*", end="")
            else:
                print("%4s " % data_set[obs][1][idx], end="")
            if (idx % 28) == 27:
                print(" ")
        print("LABEL: %s" % data_set[obs][0], end="")
        print(" ")


def main():
    show("valid.csv", "pixels")


if __name__ == "__main__":
    main()
