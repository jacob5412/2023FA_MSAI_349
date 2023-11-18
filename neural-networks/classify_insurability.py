from data_loaders.read_data import read_insurability


def classify_insurability():
    train = read_insurability("three_train.csv")
    valid = read_insurability("three_valid.csv")
    test = read_insurability("three_test.csv")

    # insert code to train simple FFNN and produce evaluation metrics
