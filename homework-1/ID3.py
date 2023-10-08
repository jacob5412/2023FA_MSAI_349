import math
from collections import Counter

from node import Node


def get_most_common_class(class_labels):
    """
    Return the class with the most number of examples.
    """
    return Counter(class_labels).most_common()[0][0]


def get_entropy(examples):
    """
    Calculate the entropy for a set of examples.
    """
    class_labels_count = Counter([example["Class"] for example in examples])
    entropy = 0
    for class_label_count in class_labels_count.items():
        proportion = class_label_count[1] / len(examples)
        entropy += -proportion * math.log2(proportion)
    return entropy


def get_info_gain_and_entropy(examples, attribute):
    """
    Return the information gain and entropy for a set of examples
    and attributes.
    """
    parent_entropy = get_entropy(examples)
    attribute_values = set(example[attribute] for example in examples)
    weighted_entropy = 0
    for attribute_value in attribute_values:
        child_examples = [
            example for example in examples if example[attribute] == attribute_value
        ]
        child_entropy = get_entropy(child_examples)
        weighted_entropy += (len(child_examples) / len(examples)) * child_entropy
    info_gain = parent_entropy - weighted_entropy
    return info_gain, weighted_entropy


def ID3(examples, default):
    """
    Takes in an array of examples, and returns a tree (an instance of Node)
    trained on the examples.  Each example is a dictionary of attribute:value
    pairs, and the target class variable is a special attribute with the name
    "Class". Any missing attributes are denoted with a value of "?".
    """
    if len(examples) == 0:
        node = Node(default)
        return node
    attributes = set(
        attribute for attribute in examples[0].keys() if attribute != "Class"
    )
    node = ID3_helper(examples, attributes)
    return node


def ID3_helper(examples, attributes, missing_values="keep"):
    """
    Recursively creates a decision tree.
    """
    node = Node()
    class_labels = [example["Class"] for example in examples]

    # this class label would be useful during pruning
    node.update_class_label(get_most_common_class(class_labels))

    # if all examples belong to the same class, update as leaf and return
    if len(set(class_labels)) == 1:
        node.update_as_leaf()
        return node

    # if no attributes remaining, update as leaf and use most common class
    if not attributes:
        node.update_as_leaf()
        return node

    # get best_attribute based on information gain (info_gain)
    attribute_info_gain_and_entropy = {}
    for attribute in attributes:
        attribute_info_gain_and_entropy[attribute] = get_info_gain_and_entropy(
            examples, attribute
        )
    best_attribute = max(
        attribute_info_gain_and_entropy,
        key=lambda key: attribute_info_gain_and_entropy[key][0],
    )
    node.update_attribute(best_attribute)
    node.update_info_gain_and_entropy(
        attribute_info_gain_and_entropy[best_attribute][0],
        attribute_info_gain_and_entropy[best_attribute][1],
    )

    # recursively create child nodes based on the best attribute values
    if missing_values == "ignore":
        best_attribute_values = set(
            example[best_attribute]
            for example in examples
            if example[best_attribute] != "?"
        )
    elif missing_values == "keep":
        best_attribute_values = set(example[best_attribute] for example in examples)
    for best_attribute_value in best_attribute_values:
        child_examples = [
            example
            for example in examples
            if example[best_attribute] == best_attribute_value
        ]
        child_attributes = set(
            attribute for attribute in attributes if attribute != best_attribute
        )
        child_node = ID3(child_examples, child_attributes)
        node.add_child(best_attribute_value, child_node)
    return node


def prune(node, examples):
    """
    Takes in a trained tree and a validation set of examples.  Prunes nodes in
    order to improve accuracy on the validation data; the precise pruning
    strategy is up to you.
    """
    accuracy_based_pruning(node, examples)


def accuracy_based_pruning(node, examples):
    """
    Recursively prune a tree by cutting off children until accuracy on the
    validation set stops improving.
    """
    if node.is_leaf:
        return

    for _, child_node in node.get_children():
        accuracy_based_pruning(child_node, examples)

    # Start pruning when recursion ends
    pre_pruning_accuracy = test(node, examples)
    node.is_leaf = True
    post_pruning_accuracy = test(node, examples)

    # only prune the tree if the accuracy is better
    if post_pruning_accuracy <= pre_pruning_accuracy:
        node.is_leaf = False


def test(node, examples):
    """
    Takes in a trained tree and a test set of examples.  Returns the accuracy
    (fraction of examples the tree classifies correctly).
    """
    # compare predicted class label with ground truth
    # TODO: Vectorize this
    num_correct_predictions = 0
    for example in examples:
        y_hat = evaluate(node, example)
        if y_hat == example["Class"]:
            num_correct_predictions += 1
    return num_correct_predictions / len(examples)  # accuracy


def evaluate(node, example):
    """
    Takes in a tree and one example. Returns the Class value that the tree
    assigns to the example.
    """
    # recursively traverse the tree, until you reach a leaf node
    if node.is_leaf:
        return node.class_label
    node_attribute = node.get_attribute()
    example_attribute_value = example.get(node_attribute)
    child_node = node.children.get(example_attribute_value)
    # if attribute value is missing or if tree is pruned
    # class_label here is the majority class
    if not child_node:
        return node.class_label
    return evaluate(child_node, example)
