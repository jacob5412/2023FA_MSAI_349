"""
Module for training and testing a neural network model on MNIST data.
"""
import torch
from torch import nn
from torch.utils.data import DataLoader

from data_loaders.mnist_data import CustomMnistDataset
from data_loaders.read_data import read_mnist
from data_loaders.standard_scaler import StandardScaler
from networks.evaluate_network import evaluate_network
from networks.mnist_q2 import FeedForward
from networks.test_network import test_network
from networks.train_network import train_network
from utils.plot_evaluation import plot_accuracy_curve, plot_learning_curve

MINIMUM_LEARNING_RATE = 1e-6
PRINT_INTERVAL = 150
BASE_PATH = "results/question_2/"


def train_and_test_q2():
    """
    Perform training and testing of a neural network model on MNIST data.
    """
    # Load training data
    train = read_mnist("data/mnist_train.csv")
    train_features = train[:, 1:]

    # Scaling data
    ss = StandardScaler()
    ss.fit(train_features)

    # Dataset loaders
    train_data = CustomMnistDataset("data/mnist_train.csv", scaler=ss)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    valid_data = CustomMnistDataset("data/mnist_valid.csv", scaler=ss)
    valid_loader = DataLoader(valid_data, batch_size=64, shuffle=True)
    test_data = CustomMnistDataset("data/mnist_test.csv", scaler=ss)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

    # Define parameters for training
    device = "cpu"
    num_epochs = 1100
    learning_rate = 0.01
    lr_decay_factor = 0.2
    lr_decay_step = 100
    original_learning_rate = learning_rate

    # Initialize FeedForward model, loss function, optimizer, and lists to track metrics
    ff = FeedForward().to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(ff.parameters(), lr=learning_rate)
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        # Train the network and validate
        train_loss, train_accuracy = train_network(
            train_loader, ff, loss_func, optimizer, device
        )
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_loss, val_accuracy = test_network(valid_loader, ff, loss_func, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Print and update learning rate decay
        if (epoch + 1) % PRINT_INTERVAL == 0:
            print(f"---Epoch [{epoch + 1}/{num_epochs}]---")
            print(f"Train Loss: {train_loss:.6f}")
            print(f"Valid Loss: {val_loss:.6f}")
            print(f"Train Accuracy: {train_accuracy:.6f}")
            print(f"Valid Accuracy: {val_accuracy:.6f}\n")
        if (epoch + 1) % lr_decay_step == 0:
            learning_rate = max(
                optimizer.param_groups[0]["lr"] * lr_decay_factor,
                MINIMUM_LEARNING_RATE,
            )
            optimizer.param_groups[0]["lr"] = learning_rate
            print(f"New LR is: {optimizer.param_groups[0]['lr']:.8f}")

    # Plot learning and accuracy curves
    plot_learning_curve(
        train_losses,
        val_losses,
        [num_epochs, original_learning_rate, lr_decay_factor, lr_decay_step],
        BASE_PATH,
    )
    plot_accuracy_curve(
        train_accuracies,
        val_accuracies,
        [num_epochs, original_learning_rate, lr_decay_factor, lr_decay_step],
        BASE_PATH,
    )

    # Evaluate Network
    (
        test_loss,
        test_accuracy,
        precision,
        recall,
        f1,
        macro_precision,
        macro_recall,
        conf_matrix,
    ) = evaluate_network(test_loader, ff, loss_func, device)
    print("---Test Results---")
    print(f"Loss: {test_loss:.6f}")
    print(f"Accuracy: {test_accuracy:.6f}")
    print(f"Loss: {test_loss:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall: {recall:.6f}")
    print(f"F1-Score: {f1:.6f}")
    print(f"Macro-Precision: {macro_precision:.6f}")
    print(f"Macro-Recall: {macro_recall:.6f}")
    print("Confusion Matrix:")
    print(conf_matrix)


if __name__ == "__main__":
    train_and_test_q2()
