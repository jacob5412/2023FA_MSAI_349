"""
Module for hyperparameter search and evaluation.
"""
import csv

import torch
from torch import nn
from torch.utils.data import DataLoader

from data_loaders.mnist_data import CustomMnistDataset
from data_loaders.read_data import read_mnist
from data_loaders.standard_scaler import StandardScaler
from networks.mnist_q2 import FeedForward
from networks.mnist_q3 import FeedForwardDropout
from networks.test_network import test_network
from networks.train_network import train_network
from utils.generate_hyperparams import get_hyperparams_q3
from utils.plot_evaluation_reg import plot_accuracy_curve, plot_learning_curve

MINIMUM_LEARNING_RATE = 1e-5
PRINT_INTERVAL = 150
BASE_PATH = "hyperparams/question_3/"


def save_to_csv(data):
    """
    Save the results data to a CSV file.

    Parameters:
    - data (list): List of results data to be saved.

    Returns:
    - None
    """
    headers = [
        "num_epochs",
        "learning_rate",
        "lr_decay_factor",
        "lr_decay_step",
        "regularization",
        "final_train_loss",
        "final_valid_loss",
    ]

    with open(BASE_PATH + "results.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(data)


def hyperparams_search_q3():
    """
    Perform hyperparameter search for a neural network model on mnist data.

    Returns:
    - None
    """
    # load training data
    train = read_mnist("mnist_train.csv")
    train_features = train[:, 1:]

    # scaling data
    ss = StandardScaler()
    ss.fit(train_features)

    # dataset loaders
    train_data = CustomMnistDataset("mnist_train.csv", scaler=ss)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    valid_data = CustomMnistDataset("mnist_valid.csv", scaler=ss)
    valid_loader = DataLoader(valid_data, batch_size=64, shuffle=True)

    hyperparams_list = get_hyperparams_q3()
    device = "cpu"
    results = []

    for hyperparams in hyperparams_list:
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        (
            num_epochs,
            learning_rate,
            lr_decay_factor,
            lr_decay_step,
            regularization,
        ) = hyperparams
        orginal_learning_rate = learning_rate

        loss_func = nn.CrossEntropyLoss()
        if regularization != 0:
            ff = FeedForward().to(device)
            optimizer = torch.optim.Adam(
                ff.parameters(), lr=learning_rate, weight_decay=regularization
            )
        else:
            ff = FeedForwardDropout().to(device)
            optimizer = torch.optim.Adam(ff.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            # fetch train and valid losses
            train_loss, train_accuracy = train_network(
                train_loader, ff, loss_func, optimizer
            )
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)

            val_loss, val_accuracy = test_network(valid_loader, ff, loss_func)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            # print loss
            if (epoch + 1) % PRINT_INTERVAL == 0:
                print(f"---Epoch [{epoch + 1}/{num_epochs}]---")
                print(f"Train Loss: {train_loss:.6f}")
                print(f"Valid Loss: {val_loss:.6f}")
                print(f"Train Accuracy: {train_accuracy:.6f}")
                print(f"Valid Accuracy: {val_accuracy:.6f}\n")

            # Learning rate decay schedule
            if (epoch + 1) % lr_decay_step == 0:
                learning_rate = max(
                    optimizer.param_groups[0]["lr"] * lr_decay_factor,
                    MINIMUM_LEARNING_RATE,
                )
                optimizer.param_groups[0]["lr"] = learning_rate
        results.append(list(hyperparams) + [train_loss, val_loss])
        plot_learning_curve(
            train_losses,
            val_losses,
            [
                num_epochs,
                orginal_learning_rate,
                lr_decay_factor,
                lr_decay_step,
                regularization,
            ],
            BASE_PATH,
        )
        plot_accuracy_curve(
            train_accuracies,
            val_accuracies,
            [
                num_epochs,
                orginal_learning_rate,
                lr_decay_factor,
                lr_decay_step,
                regularization,
            ],
            BASE_PATH,
        )
    save_to_csv(results)


if __name__ == "__main__":
    hyperparams_search_q3()
