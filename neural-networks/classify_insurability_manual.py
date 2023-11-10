from utils.read_data import read_insurability


def classify_insurability_manual():
    train = read_insurability("three_train.csv")
    valid = read_insurability("three_valid.csv")
    test = read_insurability("three_test.csv")

    # reimplement classify_insurability() without using a PyTorch optimizer.
    # this part may be simpler without using a class for the FFNN
