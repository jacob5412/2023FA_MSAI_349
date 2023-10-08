import parse
import ID3
from collections import Counter
import random

def random_forest(examples, number_of_trees, depth_of_a_tree):
    return bootstrap_data(examples, number_of_trees, depth_of_a_tree)


def bootstrap_data(examples, number_of_trees, depth_of_a_tree):
    random_forest_tree_list = list()
    # bootstrapping given dataset
    for i in range(number_of_trees):
        bootstrap_sample = list()
        row_count = 0
        while row_count < depth_of_a_tree:
            bootstrap_sample.append(examples[random.randint(0, len(examples)-1)])
            row_count += 1
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
