import torch
from torch import nn
from torch.utils.data import DataLoader

from data_loaders.mnist_data import CustomMnistDataset
from data_loaders.read_data import read_mnist
from data_loaders.standard_scaler import StandardScaler
from networks.mnist_base_nn import FeedForward
from networks.test_network import test_network
from networks.train_network import train_network


def classify_mnist():
    # load training data
    train = read_mnist("mnist_train.csv")
    train_features = train[:, 1:]

    # load validation data
    valid = read_mnist("mnist_valid.csv")

    # scaling data
    ss = StandardScaler()
    ss.fit(train_features)

    # loaders for nn
    train_data = CustomMnistDataset("mnist_train.csv", scaler=ss)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    valid_data = CustomMnistDataset("mnist_valid.csv", scaler=ss)
    valid_loader = DataLoader(valid_data, batch_size=64, shuffle=True)

    device = "cpu"
    ff = FeedForward().to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(ff.parameters(), lr=1e-3)
    epochs = 100
    train_loss = []
    test_loss = []
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        losses = train_network(train_loader, ff, loss_func, optimizer)
        train_loss.append(losses)
        test_loss.append(test_network(valid_loader, ff, loss_func))

    valid = read_mnist("mnist_valid.csv")
    test = read_mnist("mnist_test.csv")


if __name__ == "__main__":
    classify_mnist()
