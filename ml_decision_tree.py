# Machine Learning HomeWork 2
# name: Tejaswi Gorrepati
# description: Training and testing decision trees with discrete-values attributes

import sys
import math
import pandas as pd
import numpy as np

class DecisionNode:

    # A DecisionNode contains an attribute and a dictionary of children.
    # The attribute is either the attribute being split on, or the predicted label if the node has no children.
    def __init__(self, attribute):
        self.attribute = attribute
        self.children = {}

    # Visualizes the tree
    def display(self, level = 0):
        if self.children == {}: # reached leaf level
            print(": ", self.attribute, end="")
        else:
            for value in self.children.keys():
                prefix = "\n" + " " * level * 4
                print(prefix, self.attribute, "=", value, end="")
                self.children[value].display(level + 1)

    # Predicts the target label for instance x
    def predicts(self, x):
        if self.children == {}: # reached leaf level
            return self.attribute
        value = x[self.attribute]
        subtree = self.children[value]
        return subtree.predicts(x)

#This function computes entropy
def entropy(s):
    res = 0
    var, counts = np.unique(s, return_counts=True)
    #print ('var: {0} counnts: {1} set: {2}'.format(var, counts,s))
    freqs = counts.astype('float')/len(s)
    #print ('Freqs: {0}'.format(freqs))
    for p in freqs:
        if p != 0.0:
            res -= p * np.log2(p)
    return res


def partition(a):
    #print(a)
    #for c in np.unique(a):
    #    print(c)
    #    print ((a==c).nonzero()[0])
    return {c: (a==c).nonzero()[0] for c in np.unique(a)}

# Calculate information gain for each attribute split
def information_gain(y, x):
    res = entropy(y)
    val, counts = np.unique(x, return_counts=True)
    freqs = counts.astype('float')/len(x)
    for p, v in zip(freqs, val):
        res -= p * entropy(y[x == v])
    return res

# Recursive function to create decison tree based on the training data
def decison_tree(x, y, headers):

    if len(set(y)) == 1:
        return DecisionNode(y[0])

    # Get the attribute that gives the highest gain
    gain = np.array([information_gain(y, x_attr) for x_attr in x.T])
    selected_attr = np.argmax(gain)   # returns index of the max gain
    #print ('Selection attribute: ' + str(selected_attr))

    # If there's no gain at all, nothing has to be done, just return the node
    if np.all(gain < 1e-100):
        return DecisionNode(y[0])

    # Splitting using selected attribute
    sets = partition(x[:, selected_attr])
    #print (x[:, selected_attr])
    #print ('sets: ' + str(sets))

    node = DecisionNode(headers[selected_attr])
    for k, v in sets.items():
        y_subset = y.take(v, axis=0)
        x_subset = x.take(v, axis=0)
        node.children[k] = decison_tree(x_subset, y_subset, headers)

    return node

def id3(examples, target, attributes):

    # Preparing data for creating decision tree
    y_train = examples.loc[:, target]
    x_train = examples.drop(target, axis=1)
    x = x_train.as_matrix()
    y = y_train.as_matrix().T

    # Create decison tree
    tree = decison_tree(x,y,attributes)
    return tree


####################   MAIN PROGRAM ######################

# Reading input data
train = pd.read_csv(sys.argv[1])
test = pd.read_csv(sys.argv[2])
target = sys.argv[3]
attributes = train.columns.tolist()
attributes.remove(target)

# Learning and visualizing the tree
tree = id3(train,target,attributes)
tree.display()

# Evaluating the tree on the test data
correct = 0
for i in range(0,len(test)):
    if str(tree.predicts(test.loc[i])) == str(test.loc[i,target]):
        correct += 1
print("\nThe accuracy is: ", correct/len(test))
