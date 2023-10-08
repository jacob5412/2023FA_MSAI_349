import math
import random

import ID3
import matplotlib.pyplot as plt
import parse

def plot_accuracy(
    training_sizes,
    avg_accuracies_with_pruning,
    avg_accuracies_without_pruning,
    save_path,
    learning_curve_title
):
    """
    Plot learning curves.

    Args:
        training_sizes: List of training set sizes
        avg_accuracies_with_pruning: List of average accuracies with pruning
        avg_accuracies_without_pruning: List of average accuracies
                                        without pruning
    """
    plt.figure(figsize=(10, 6))
    plt.plot(training_sizes, avg_accuracies_with_pruning, label="With Pruning")
    plt.plot(training_sizes, avg_accuracies_without_pruning, label="Without Pruning")
    plt.xlabel("Number of Training Examples")
    plt.ylabel("Average Accuracy on Test Data")
    plt.title(f"Learning Curves for {learning_curve_title} With and Without Pruning")
    plt.legend()
    plt.grid(True)
    # save the plot as an image if save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    print(f"saved image to {save_path}")


# inFile - string location of the house data file
def testPruningOnHouseData(inFile):
    save_path_house = "./images/learning_curve_house.png"
    data = parse.parse(inFile)
    avg_test_accuracy_with_pruning = []
    avg_test_accuracy_without_pruning = []
    training_sizes = range(
        10, 305, 5
    )  # Adjusted to handle smaller datasets

    print("Calculating ...")

    for train_size in training_sizes:
        withPruning = []
        withoutPruning = []
        data = parse.parse(inFile)
        for _ in range(100):
            random.shuffle(data)  # Cross validated-"mention in pdf"
            # Calculate the sizes for validation, and test sets
            validation_size = math.ceil((int(train_size / 0.8) - train_size) // 2)
            test_size = validation_size

            train = data[:train_size]
            valid = data[train_size : train_size + validation_size]
            test = data[
                train_size + validation_size : train_size + validation_size + test_size
            ]

            tree = ID3.ID3(train, 0)
            ID3.prune(tree, valid)
            acc = ID3.test(tree, test)
            withPruning.append(acc)
            tree = ID3.ID3(train + valid, 0)
            acc = ID3.test(tree, test)
            withoutPruning.append(acc)

        avg_accuracy_with_pruning = sum(withPruning) / len(withPruning)
        avg_accuracy_without_pruning = sum(withoutPruning) / len(withoutPruning)

        avg_test_accuracy_with_pruning.append(avg_accuracy_with_pruning)
        avg_test_accuracy_without_pruning.append(avg_accuracy_without_pruning)

    print("Done.")

    plot_accuracy(
        training_sizes,
        avg_test_accuracy_with_pruning,
        avg_test_accuracy_without_pruning,
        save_path_house,
        "House Votes Data"
    )


# inFile - string location of the house data file
def testPruningOnTennisData(inFile):
    save_path_tennis = "./images/learning_curve_tennis.png"
    data = parse.parse(inFile)
    avg_test_accuracy_with_pruning = []
    avg_test_accuracy_without_pruning = []
    training_sizes = range(
        1, 9, 1
    )  # Adjusted to handle smaller datasets

    print("Calculating ...")

    for train_size in training_sizes:
        withPruning = []
        withoutPruning = []
        data = parse.parse(inFile)
        for _ in range(10):
            random.shuffle(data)  # Cross validated-"mention in pdf"
            # Calculate the sizes for validation, and test sets
            validation_size = max(
                1, math.ceil((int(train_size / 0.5) - train_size) // 2)
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
            withPruning.append(acc)
            tree = ID3.ID3(train + valid, 0)
            acc = ID3.test(tree, test)
            withoutPruning.append(acc)

        avg_accuracy_with_pruning = sum(withPruning) / len(withPruning)
        avg_accuracy_without_pruning = sum(withoutPruning) / len(withoutPruning)

        avg_test_accuracy_with_pruning.append(avg_accuracy_with_pruning)
        avg_test_accuracy_without_pruning.append(avg_accuracy_without_pruning)

    print("Done.")

    plot_accuracy(
        training_sizes,
        avg_test_accuracy_with_pruning,
        avg_test_accuracy_without_pruning,
        save_path_tennis,
        "Tennis Data"
    )


# inFile - string location of the house data file
def testPruningOnCandyData(inFile):
    save_path_candy = "./images/learning_curve_candy.png"
    data = parse.parse(inFile)
    avg_test_accuracy_with_pruning = []
    avg_test_accuracy_without_pruning = []
    training_sizes = range(
        5, 64, 4
    )  # Adjusted to handle smaller datasets

    print("Calculating ...")

    for train_size in training_sizes:
        withPruning = []
        withoutPruning = []
        data = parse.parse(inFile)
        for _ in range(25):
            random.shuffle(data)  # Cross validated-"mention in pdf"
            # Calculate the sizes for validation, and test sets
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
            withPruning.append(acc)
            tree = ID3.ID3(train + valid, 0)
            acc = ID3.test(tree, test)
            withoutPruning.append(acc)

        avg_accuracy_with_pruning = sum(withPruning) / len(withPruning)
        avg_accuracy_without_pruning = sum(withoutPruning) / len(withoutPruning)

        avg_test_accuracy_with_pruning.append(avg_accuracy_with_pruning)
        avg_test_accuracy_without_pruning.append(avg_accuracy_without_pruning)

    print("Done.")

    plot_accuracy(
        training_sizes,
        avg_test_accuracy_with_pruning,
        avg_test_accuracy_without_pruning,
        save_path_candy,
        "Candy Data"

    )


if __name__ == "__main__":
    testPruningOnHouseData(inFile="house_votes_84.data")
    testPruningOnTennisData(inFile="tennis.data")
    testPruningOnCandyData(inFile="candy.data")
