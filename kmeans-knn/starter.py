# returns a list of labels for the query dataset based upon labeled observations in the train dataset.
# metric is a string specifying either "euclidean" or "cosim".
# All hyper-parameters should be hard-coded in the algorithm.
def knn(train, query, metric):
    return labels


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
