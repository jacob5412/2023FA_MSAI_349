from utils.read_data import read_mnist


def classify_mnist():
    train = read_mnist("mnist_train.csv")
    valid = read_mnist("mnist_valid.csv")
    test = read_mnist("mnist_test.csv")
    show_mnist("mnist_test.csv", "pixels")

    # insert code to train a neural network with an architecture of your choice
    # (a FFNN is fine) and produce evaluation metrics
