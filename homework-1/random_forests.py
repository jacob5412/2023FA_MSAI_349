"""
Implementation of the random forests algorithm.
"""
import random

import ID3


class RandomForests:
    """
    Random Forests - an ensemble learning technique that builds multiple trees.
    Here, we're building trees using bootstrapping (sampling with replacement)

    Attributes:
        random_forests_nodes (list): A list to store the decision tree nodes
                                     in the random forest.
        num_trees (int): The number of decision trees in the random forest.
    """

    def __init__(self, num_trees):
        self.random_forests_nodes = []
        self.num_trees = num_trees

    def fit(self, examples, default):
        """
        Fits the random forest to a dataset using bootstrapped samples and
        creates decision trees.
        """
        for _ in range(self.num_trees):
            # create a boostrapped sample by randomly selecting an example n
            # times; with replacement, i.e., can select one example more than
            # once; n = len(examples)
            bootstrap_sample = [random.choice(examples) for _ in range(len(examples))]
            random_forests_node = ID3.ID3(bootstrap_sample, default)
            self.random_forests_nodes.append(random_forests_node)

    def test(self, examples):
        """
        Tests the accuracy of the random forest on a dataset.
        """
        num_correct_predictions = sum(
            [self.evaluate(example) == example["Class"] for example in examples]
        )
        return num_correct_predictions / len(examples)

    def evaluate(self, example):
        """
        Evaluates a single example using the random forest's ensemble of
        decision trees.
        """
        predictions = [
            ID3.evaluate(random_forests_node, example)
            for random_forests_node in self.random_forests_nodes
        ]
        return ID3.get_most_common_class(predictions)
