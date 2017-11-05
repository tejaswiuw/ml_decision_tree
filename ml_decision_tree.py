# course: TCSS555
# Homework 2
# date: 10/03/2017
# name: Martine De Cock
# description: Training and testing decision trees with discrete-values attributes

import sys
import math
import pandas as pd

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


# Illustration of functionality of DecisionNode class
def funTree():


    myLeftTree = DecisionNode('humidity')
    myLeftTree.children['normal'] = DecisionNode('no')
    myLeftTree.children['high'] = DecisionNode('yes')
    myTree = DecisionNode('wind')
    myTree.children['weak'] = myLeftTree
    myTree.children['strong'] = DecisionNode('no')
    return myTree

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

#Method is called while fitting the training data
def recursiveTree(x, y, headers):
    if len(set(y)) == 1:
        return y[0]

    # We get attribute that gives the highest mutual
    gain = np.array([information_gain(y, x_attr) for x_attr in x.T])
    selected_attr = np.argmax(gain)   # returns index of the max gain
    #print ('Selection attribute: ' + str(selected_attr))

    # If there's no gain at all, nothing has to be done, just return the original set
    if np.all(gain < 1e-100):
        #print("@@@@@@@@@This Leaf Node does not give unique result@@@@@@@@@@")
        # Tree could no further split data and has no unique class at leaf
        return y

    # Splitting using selected attribute
    print (x[:, selected_attr])
    sets = partition(x[:, selected_attr])
    #print ('sets: ' + str(sets))
    res = {}
    for k, v in sets.items():
        y_subset = y.take(v, axis=0)
        x_subset = x.take(v, axis=0)
        # print (x_subset)
        res["%s = %s" % (headers[selected_attr], k)] = recursiveTree(x_subset, y_subset, headers)

    return res



def id3(examples, target, attributes):
    tree = funTree()
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
