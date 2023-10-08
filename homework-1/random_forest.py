import numpy as np
import parse
import ID3
from collections import Counter


def random_forest(examples, number_of_trees, depth_of_a_tree):
    return bootstrap_data(examples, number_of_trees, depth_of_a_tree)


def bootstrap_data(examples, number_of_trees, depth_of_a_tree):
    random_forest_tree_list = list()
    # bootstrapping given dataset
    for i in range(number_of_trees):
        bootstrap_sample = np.random.choice(examples, size=depth_of_a_tree, replace=True)
        random_forest_tree = ID3.ID3(bootstrap_sample, 0)
        random_forest_tree_list.append(random_forest_tree)
    return random_forest_tree_list


def aggregation(random_forest_tree_list, random_features):
    accuracy_votes = list()
    for random_forest_tree in random_forest_tree_list:
        accuracy_vote = ID3.test(random_forest_tree, random_features)
        accuracy_votes.append(accuracy_vote)
    return Counter(accuracy_votes).most_common(1)[0][0]


def test(examples):
    number_of_trees = 5
    depth_of_a_tree = len(examples)
    random_forest_tree_list = random_forest(examples, number_of_trees, depth_of_a_tree)

    # TODO: Review this: as the root node is dynamically changing for random forest trees, the features/columns cannot be randomized
    feature_selction_accuray = aggregation(random_forest_tree_list, examples)
    print(f"Feature selection accuracy %d" % feature_selction_accuray)


test(parse.parse("candy.data"))
