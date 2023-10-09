"""
Unit Tests
"""
import random

import ID3
import parse
from random_forest import RandomForest


def testID3AndEvaluate():
    """
    Test the ID3 algorithm and evaluate its correctness.
    """
    data = [dict(a=1, b=0, Class=1), dict(a=1, b=1, Class=1)]
    tree = ID3.ID3(data, 0)
    if tree != None:
        ans = ID3.evaluate(tree, dict(a=1, b=0))
        if ans != 1:
            print("ID3 test failed.")
        else:
            print("ID3 test succeeded.")
    else:
        print("ID3 test failed -- no tree returned")


def testPruning():
    """
    Test the pruning of an ID3 decision tree.
    """
    data = [
        dict(a=0, b=1, c=1, d=0, Class=1),
        dict(a=0, b=0, c=1, d=0, Class=0),
        dict(a=0, b=1, c=0, d=0, Class=1),
        dict(a=1, b=0, c=1, d=0, Class=0),
        dict(a=1, b=1, c=0, d=0, Class=0),
        dict(a=1, b=1, c=0, d=1, Class=0),
        dict(a=1, b=1, c=1, d=0, Class=0),
    ]
    validationData = [
        dict(a=0, b=0, c=1, d=0, Class=1),
        dict(a=1, b=1, c=1, d=1, Class=0),
    ]
    tree = ID3.ID3(data, 0)
    ID3.prune(tree, validationData)
    if tree != None:
        ans = ID3.evaluate(tree, dict(a=0, b=0, c=1, d=0))
        if ans != 1:
            print("pruning test failed.")
        else:
            print("pruning test succeeded.")
    else:
        print("pruning test failed -- no tree returned.")


def testID3AndTest():
    """
    Test the ID3 algorithm's accuracy on training and test data.
    """
    trainData = [
        dict(a=1, b=0, c=0, Class=1),
        dict(a=1, b=1, c=0, Class=1),
        dict(a=0, b=0, c=0, Class=0),
        dict(a=0, b=1, c=0, Class=1),
    ]
    testData = [
        dict(a=1, b=0, c=1, Class=1),
        dict(a=1, b=1, c=1, Class=1),
        dict(a=0, b=0, c=1, Class=0),
        dict(a=0, b=1, c=1, Class=0),
    ]
    tree = ID3.ID3(trainData, 0)
    fails = 0
    if tree != None:
        acc = ID3.test(tree, trainData)
        if acc == 1.0:
            print("testing on train data succeeded.")
        else:
            print("testing on train data failed.")
            fails = fails + 1
        acc = ID3.test(tree, testData)
        if acc == 0.75:
            print("testing on test data succeeded.")
        else:
            print("testing on test data failed.")
            fails = fails + 1
        if fails > 0:
            print("Failures: ", fails)
        else:
            print("testID3AndTest succeeded.")
    else:
        print("testID3andTest failed -- no tree returned.")


# inFile - string location of the house data file
def testPruningOnHouseData(inFile):
    """
    Test pruning of ID3 on house data.
    """
    withPruning = []
    withoutPruning = []
    data = parse.parse(inFile)
    for _ in range(100):
        random.shuffle(data)
        train = data[: len(data) // 2]
        valid = data[len(data) // 2 : 3 * len(data) // 4]
        test = data[3 * len(data) // 4 :]

        tree = ID3.ID3(train, "democrat")
        acc = ID3.test(tree, train)
        print("training accuracy: ", acc)
        acc = ID3.test(tree, valid)
        print("validation accuracy: ", acc)
        acc = ID3.test(tree, test)
        print("test accuracy: ", acc)

        ID3.prune(tree, valid)
        acc = ID3.test(tree, train)
        print("pruned tree train accuracy: ", acc)
        acc = ID3.test(tree, valid)
        print("pruned tree validation accuracy: ", acc)
        acc = ID3.test(tree, test)
        print("pruned tree test accuracy: ", acc)
        withPruning.append(acc)
        tree = ID3.ID3(train + valid, "democrat")
        acc = ID3.test(tree, test)
        print("no pruning test accuracy: ", acc)
        withoutPruning.append(acc)

    print(
        "average with pruning",
        sum(withPruning) / len(withPruning),
        " without: ",
        sum(withoutPruning) / len(withoutPruning),
    )


def testPruningCandyData(inFile):
    """
    Test pruning of ID3 on candy data.
    """
    withPruning = []
    withoutPruning = []
    data = parse.parse(inFile)
    for _ in range(25):
        random.shuffle(data)
        train = data[: len(data) // 2]
        valid = data[len(data) // 2 : 3 * len(data) // 4]
        test = data[3 * len(data) // 4 :]

        tree = ID3.ID3(train, 0)
        acc = ID3.test(tree, train)
        print("training accuracy: ", acc)
        acc = ID3.test(tree, valid)
        print("validation accuracy: ", acc)
        acc = ID3.test(tree, test)
        print("test accuracy: ", acc)

        ID3.prune(tree, valid)
        acc = ID3.test(tree, train)
        print("pruned tree train accuracy: ", acc)
        acc = ID3.test(tree, valid)
        print("pruned tree validation accuracy: ", acc)
        acc = ID3.test(tree, test)
        print("pruned tree test accuracy: ", acc)
        withPruning.append(acc)
        tree = ID3.ID3(train + valid, "democrat")
        acc = ID3.test(tree, test)
        print("no pruning test accuracy: ", acc)
        withoutPruning.append(acc)

    print(
        "average with pruning",
        sum(withPruning) / len(withPruning),
        " without: ",
        sum(withoutPruning) / len(withoutPruning),
    )


def testPruningTennisData(inFile):
    """
    Test pruning of ID3 on tennis data.
    """
    withPruning = []
    withoutPruning = []
    data = parse.parse(inFile)
    for _ in range(10):
        random.shuffle(data)
        train = data[: len(data) // 2]
        valid = data[len(data) // 2 : 3 * len(data) // 4]
        test = data[3 * len(data) // 4 :]

        tree = ID3.ID3(train, 0)
        acc = ID3.test(tree, train)
        print("training accuracy: ", acc)
        acc = ID3.test(tree, valid)
        print("validation accuracy: ", acc)
        acc = ID3.test(tree, test)
        print("test accuracy: ", acc)

        ID3.prune(tree, valid)
        acc = ID3.test(tree, train)
        print("pruned tree train accuracy: ", acc)
        acc = ID3.test(tree, valid)
        print("pruned tree validation accuracy: ", acc)
        acc = ID3.test(tree, test)
        print("pruned tree test accuracy: ", acc)
        withPruning.append(acc)
        tree = ID3.ID3(train + valid, "democrat")
        acc = ID3.test(tree, test)
        print("no pruning test accuracy: ", acc)
        withoutPruning.append(acc)

    print(
        "average with pruning",
        sum(withPruning) / len(withPruning),
        " without: ",
        sum(withoutPruning) / len(withoutPruning),
    )


def testRandomForestOnHouseData(inFile):
    """
    Test the Random Forest algorithm on a dataset of house data.
    """
    print("\nRandom Forest on House Data:")
    data = parse.parse(inFile)
    train = data[: len(data) // 2]
    valid = data[len(data) // 2 : 3 * len(data) // 4]
    test = data[3 * len(data) // 4 :]
    random_forest = RandomForest(num_trees=5)
    random_forest.fit(train, "democrat")
    acc = random_forest.test(train)
    print("training accuracy: ", acc)
    acc = random_forest.test(valid)
    print("validation accuracy: ", acc)
    acc = random_forest.test(test)
    print("testing accuracy: ", acc)


if __name__ == "__main__":
    testID3AndEvaluate()
    testPruning()
    testID3AndTest()
    testPruningOnHouseData(inFile="house_votes_84.data")

    # additional tests
    testRandomForestOnHouseData(inFile="house_votes_84.data")
    testPruningCandyData("candy.data")
    testPruningTennisData("tennis.data")
