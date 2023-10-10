"""
Plotting the learning curves for house data, candy data, and
tennis data.
"""
import math
import random

import ID3
import matplotlib.pyplot as plt
import parse


def plot_learning_curve(
    training_sizes,
    avg_accuracies_with_pruning,
    avg_accuracies_without_pruning,
    dataset_name,
):
    """
    Plot learning curves.

    Args:
        training_sizes: List of training set sizes
        avg_accuracies_with_pruning: List of average accuracies with pruning
        avg_accuracies_without_pruning: List of average accuracies
                                        without pruning
        dataset_name: name of the dataset of plot title and image filename.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(
        training_sizes,
        avg_accuracies_with_pruning,
        label="With Pruning",
        color="orange",
        marker="o",
    )
    plt.plot(
        training_sizes,
        avg_accuracies_without_pruning,
        label="Without Pruning",
        color="blue",
        marker="o",
    )
    plt.xlabel("Number of Training Examples")
    plt.ylabel("Average Accuracy on Test Data")
    plt.title(f"Learning Curves for {dataset_name} Data With and Without Pruning")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"images/learning_curve_{dataset_name}.png", bbox_inches="tight")
    print(f"Saved image as images/learning_curve_{dataset_name}.png")


def calculate_accuracy_for_plot(
    dataset_filename, training_sizes, loop_size, dataset_name
):
    data = parse.parse(dataset_filename)
    avg_test_accuracy_with_pruning = []
    avg_test_accuracy_without_pruning = []

    for train_size in training_sizes:
        with_pruning = []
        without_pruning = []

        for _ in range(loop_size):
            random.shuffle(data)
            validation_size = max(
                1, math.ceil((int(train_size / 0.8) - train_size) // 2)
            )
            test_size = validation_size
            train = data[:train_size]
            valid = data[train_size : train_size + validation_size]
            test = data[
                train_size + validation_size : train_size + validation_size + test_size
            ]

            tree = ID3.ID3(train, 0)
            ID3.prune(tree, valid)
            acc = ID3.test(tree, test)
            with_pruning.append(acc)
            tree = ID3.ID3(train + valid, 0)
            acc = ID3.test(tree, test)
            without_pruning.append(acc)

        avg_accuracy_with_pruning = sum(with_pruning) / len(with_pruning)
        avg_accuracy_without_pruning = sum(without_pruning) / len(without_pruning)
        avg_test_accuracy_with_pruning.append(avg_accuracy_with_pruning)
        avg_test_accuracy_without_pruning.append(avg_accuracy_without_pruning)

    plot_learning_curve(
        training_sizes,
        avg_test_accuracy_with_pruning,
        avg_test_accuracy_without_pruning,
        dataset_name,
    )


def plot_learning_curve_house_data(dataset_filename):
    training_sizes = range(10, 305, 5)
    print("Calculating ...")
    calculate_accuracy_for_plot(dataset_filename, training_sizes, 100, "House Votes")


def plot_learning_curve_tennis_data(dataset_filename):
    training_sizes = range(1, 9, 1)
    print("Calculating ...")
    calculate_accuracy_for_plot(dataset_filename, training_sizes, 10, "Tennis")


def plot_learning_curve_candy_data(dataset_filename):
    training_sizes = range(5, 64, 4)
    print("Calculating ...")
    calculate_accuracy_for_plot(dataset_filename, training_sizes, 25, "Candy")


if __name__ == "__main__":
    random.seed(101)
    plot_learning_curve_house_data(dataset_filename="house_votes_84.data")
    plot_learning_curve_tennis_data(dataset_filename="tennis.data")
    plot_learning_curve_candy_data(dataset_filename="candy.data")
