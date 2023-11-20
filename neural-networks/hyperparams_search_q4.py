"""
Module for hyperparameter search and evaluation.
"""
import csv

from torch import nn
from torch.utils.data import DataLoader

from data_loaders.insurability_data import CustomInsurabilityDataset
from data_loaders.read_data import read_insurability
from data_loaders.standard_scaler import StandardScaler
from networks.insurability_q4 import FeedForward
from networks.test_network import test_network
from networks.train_network_no_optimizer import train_network
from utils.generate_hyperparams import get_hyperparams_q4
from utils.plot_evaluation_no_optimizer import plot_accuracy_curve, plot_learning_curve

PRINT_INTERVAL = 250
BASE_PATH = "hyperparams/question_4/"


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
        "final_train_loss",
        "final_valid_loss",
        "final_train_acc",
        "final_valid_acc",
    ]

    with open(BASE_PATH + "results.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(data)


def hyperparams_search_q4():
    """
    Perform hyperparameter search for a neural network model on insurability data.

    Returns:
    - None
    """
    # load training data
    train = read_insurability("three_train.csv")
    train_features = train[:, 1:]

    # scaling data
    ss = StandardScaler()
    ss.fit(train_features)

    # dataset loaders
    train_data = CustomInsurabilityDataset("three_train.csv", scaler=ss)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    valid_data = CustomInsurabilityDataset("three_valid.csv", scaler=ss)
    valid_loader = DataLoader(valid_data, batch_size=64, shuffle=True)

    hyperparams_list = get_hyperparams_q4()
    device = "cpu"
    results = []

    for hyperparams in hyperparams_list:
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        num_epochs, learning_rate = hyperparams
        orginal_learning_rate = learning_rate

        ff = FeedForward().to(device)
        loss_func = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            # fetch train and valid losses
            train_loss, train_accuracy = train_network(
                train_loader, ff, loss_func, device, learning_rate
            )
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)

            val_loss, val_accuracy = test_network(valid_loader, ff, loss_func, device)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            # print loss
            if (epoch + 1) % PRINT_INTERVAL == 0:
                print(f"---Epoch [{epoch + 1}/{num_epochs}]---")
                print(f"Train Loss: {train_loss:.6f}")
                print(f"Valid Loss: {val_loss:.6f}")
                print(f"Train Accuracy: {train_accuracy:.6f}")
                print(f"Valid Accuracy: {val_accuracy:.6f}\n")

        results.append(
            list(hyperparams) + [train_loss, val_loss, train_accuracy, val_accuracy]
        )
        plot_learning_curve(
            train_losses,
            val_losses,
            [num_epochs, orginal_learning_rate],
            BASE_PATH,
        )
        plot_accuracy_curve(
            train_accuracies,
            val_accuracies,
            [num_epochs, orginal_learning_rate],
            BASE_PATH,
        )
    save_to_csv(results)


if __name__ == "__main__":
    hyperparams_search_q4()
