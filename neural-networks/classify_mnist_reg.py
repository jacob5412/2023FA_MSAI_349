from utils.read_data import read_mnist


def classify_mnist_reg():
    train = read_mnist("mnist_train.csv")
    valid = read_mnist("mnist_valid.csv")
    test = read_mnist("mnist_test.csv")
    show_mnist("mnist_test.csv", "pixels")

    # add a regularizer of your choice to classify_mnist()
