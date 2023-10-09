# MSAI-349: Machine Learning - Homework Assignments

For our homework assignment, we first look at the .data files:

1. candy.data: It contains the data regarding the various candies, used for random_forest
2. tennis.data
3. house_votes_84.data: It contains the data regarding registered and non- registered voters.

We then run the mini_auto_grader.py file in order to test out the code we've written; we first test it out against the ID3.py file which runs the implementation of the ID3 algorithm. Since the code is being tested on an unseen dataset with random training, we use tennis.data in order to help us test it out on a smaller dataset.

Correspondingly, we can learn plot_learning_curves.py in order to plot the learning curves for house data, candy data, and
tennis data.

We are generating random numbers and plotting the learning curve based on the pruned ID3 data which is in turn used to plot the curves based on house_votes_84, tennis, candy.data files. This, in turn, is stored in the 'Images' folder. 

For the random_forest problem, we run the tune_random_forest.py as unit tests to test out the random_forest.py which fits the random forest to a dataset using bootstrapped samples and creates decision trees. The Random Forest classifier runs on the candy.data dataset and then correspondingly compare results of the single decision tree constructed by the ID3 algorithm.
