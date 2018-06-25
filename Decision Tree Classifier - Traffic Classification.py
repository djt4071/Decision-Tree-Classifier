
# coding: utf-8

# In[39]:


#Importing Python Machine Learning Libraries
import numpy as np
np.set_printoptions(threshold=np.inf) ###temporary print options so that the output is not truncated.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree


# In[40]:


# Data Import
###seperate columns by , and apply the header
df= pd.read_csv('netmate_out.csv', sep = ',', header = None) 

df.columns =["srcip", #0
            "srcport", #1
            "dstip", #2
            "dstport", #3
            "proto", #4
            "total_fpackets", #5
            "total_fvolume", #6
            "total_bpackets", #7
            "total_bvolume", #8
            "min_fpktl", #9
            "mean_fpktl", #10
            "max_fpktl", #11
            "std_fpktl", #12
            "min_bpktl", #13
            "mean_bpktl", #14
            "max_bpktl", #15
            "std_bpktl", #16
            "min_fiat", #17
            "mean_fiat", #18
            "max_fiat", #19
            "std_fiat", #20
            "min_biat", #21
            "mean_biat", #22
            "max_biat", #23
            "std_biat", #24
            "duration", #25
            "min_active", #26
            "mean_active", #27
            "max_active", #28
            "std_active", #29
            "min_idle", #30
            "mean_idle", #31
            "max_idle", #32
            "std_idle", #33
            "sflow_fpackets", #34
            "sflow_fbytes", #35
            "sflow_bpackets", #36
            "sflow_bbytes", #37
            "fpsh_cnt", #38
            "bpsh_cnt", #39
            "furg_cnt", #40
            "burg_cnt", #41
            "total_fhlen", #42
            "total_bhlen", #43 ###44 flow statistics total
            "target_variable"] #44

interesting_features = df.loc[:, 
                           ['proto', #4 ==X0
                           'min_fpktl', #9 ==X1
                           'mean_fpktl', #10 ==X2
                           'max_fpktl', #11 ==X3
                           'std_fpktl', #12 ==X4
                           'min_bpktl', #13 ==X5
                           'mean_bpktl', #14 ==X6
                           'max_bpktl', #15 ==X7 
                           'std_bpktl', #16 ==X8
                           'min_fiat', #17 ==X9
                           'mean_fiat', #18 ==X10
                           'max_fiat', #19 ==X11
                           'std_fiat', #20 ==X12
                           'min_biat', #21 ==X13
                           'mean_biat', #22 ==X14
                           'max_biat', #23 ==X15
                           'std_biat', #24 ==X16
                           'target_variable']] #44 ==Y          


# In[41]:


header =list(interesting_features.columns.values)
#print(header)


# In[42]:


#print(interesting_features)


# In[43]:


###check the data types for the columns in the dataframe
#interesting_features.dtypes


# In[44]:


###get additional dataframe information if required.
#interesting_features.info()


# In[45]:


def unique_vals(rows, col):
    """Find the unique values for a column in a dataset."""
    return set([row[col] for row in rows])


# In[46]:


unique_vals(interesting_features.values, 6)


# In[47]:


def class_counts(rows):
    """Counts the number of each type of example in a dataset."""
    counts = {}  # a dictionary of label -> count.
    for row in rows:
        # in our dataset format, the label is always the last column
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


# In[48]:


class_counts(interesting_features.values)


# In[49]:


def is_numeric(value):
    """Test if a value is numeric."""
    return isinstance(value, int) or isinstance(value, float)


# In[50]:


is_numeric(7)


# In[51]:


class Question:
    """A Question is used to partition a dataset.

    This class just records a 'column number' (e.g., 0 for Color) and a
    'column value' (e.g., Green). The 'match' method is used to compare
    the feature value in an example to the feature value stored in the
    question. See the demo below.
    """

    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        # Compare the feature value in an example to the
        # feature value in this question.
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        # This is just a helper method to print
        # the question in a readable format.
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (
            header[self.column], condition, str(self.value))


# In[52]:


Question(0, 17)


# In[53]:


q = Question(0, 17)
q


# In[54]:


# Let's pick an example from the training set...
example = interesting_features.iloc[200]
print(example)
# ... and see if it matches the question
q.match(example) # this will be true, since the first example is Green.
#######


# In[55]:


def partition(rows, question):
    """Partitions a dataset.

    For each row in the dataset, check if it matches the question. If
    so, add it to 'true rows', otherwise, add it to 'false rows'.
    """
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


# In[56]:


###partition the data based on a question condition being true or false.
true_rows, false_rows = partition(interesting_features.values, Question(0, 17))
###rows that meet the question condition true
#true_rows
###rows that meet the question condition false
#false_rows


# In[57]:


def gini(rows):
    """Calculate the Gini Impurity for a list of rows.

    There are a few different ways to do this, I thought this one was
    the most concise. See:
    https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
    """
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity


# In[58]:


#######
# Demo:
# Let's look at some example to understand how Gini Impurity works.
#
# First, we'll look at a dataset with no mixing.
no_mixing = [['Apple'],
              ['Apple']]
# this will return 0
gini(no_mixing)


# In[59]:


# Now, we'll look at dataset with a 50:50 apples:oranges ratio
some_mixing = [['Apple'],
               ['Orange']]
# this will return 0.5 - meaning, there's a 50% chance of misclassifying
# a random example we draw from the dataset.
gini(some_mixing)


# In[60]:


# Now, we'll look at a dataset with many different labels
lots_of_mixing = [['Apple'],
                  ['Orange'],
                  ['Grape'],
                  ['Grapefruit'],
                  ['Blueberry']]
# This will return 0.8
gini(lots_of_mixing)
#######


# In[61]:


possible_feature = class_counts(interesting_features.values)
gini(possible_feature)


# In[62]:


list_features =[['DNS'],
                ['HTTP'],
                ['HTTPS'],
                ['OpenVPN']]

gini(list_features)


# In[63]:


another_list_features = [interesting_features.values[:,-1]]
#print(another_list_features)
gini(another_list_features)


# In[64]:


def info_gain(left, right, current_uncertainty):
    """Information Gain.

    The uncertainty of the starting node, minus the weighted impurity of
    two child nodes.
    """
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)


# In[65]:


#######
# Demo:
# Calculate the uncertainy of our training data.
current_uncertainty = gini(interesting_features)
current_uncertainty


# In[66]:


# How much information do we gain by partioning on 'Green'?
true_rows, false_rows = partition(interesting_features, Question(-1, 'openvpn'))
info_gain(true_rows, false_rows, current_uncertainty)


# In[67]:


# What about if we partioned on 'Red' instead?
true_rows, false_rows = partition(interesting_features, Question(0, 17))
info_gain(true_rows, false_rows, current_uncertainty)


# In[68]:


# It looks like we learned more using 'Red' (0.37), than 'Green' (0.14).
# Why? Look at the different splits that result, and see which one
# looks more 'unmixed' to you.
true_rows, false_rows = partition(interesting_features.values, Question(-1,'openvpn'))

# Here, the true_rows contain only 'Grapes'.
true_rows

false_rows


# In[69]:


# On the other hand, partitioning by Green doesn't help so much.
true_rows, false_rows = partition(interesting_features, Question(0,'Green'))

# We've isolated one apple in the true rows.
true_rows

false_rows


# In[70]:


def find_best_split(rows):
    """Find the best question to ask by iterating over every feature / value
    and calculating the information gain."""
    best_gain = 0  # keep track of the best information gain
    best_question = None  # keep train of the feature / value that produced it
    current_uncertainty = gini(rows)
    n_features = len(rows[0]) - 1  # number of columns

    for col in range(n_features):  # for each feature

        values = set([row[col] for row in rows])  # unique values in the column

        for val in values:  # for each value

            question = Question(col, val)

            # try splitting the dataset
            true_rows, false_rows = partition(rows, question)

            # Skip this split if it doesn't divide the
            # dataset.
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            # Calculate the information gain from this split
            gain = info_gain(true_rows, false_rows, current_uncertainty)

            # You actually can use '>' instead of '>=' here
            # but I wanted the tree to look a certain way for our
            # toy dataset.
            if gain >= best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question


# In[71]:


#######
# Demo:
# Find the best question to ask first for our toy dataset.
best_gain, best_question = find_best_split(interesting_features.values)
best_question
# FYI: is color == Red is just as good. See the note in the code above
# where I used '>='.
#######


# In[72]:


class Leaf:
    """A Leaf node classifies data.

    This holds a dictionary of class (e.g., "Apple") -> number of times
    it appears in the rows from the training data that reach this leaf.
    """

    def __init__(self, rows):
        self.predictions = class_counts(rows)


# In[73]:


class Decision_Node:
    """A Decision Node asks a question.

    This holds a reference to the question, and to the two child nodes.
    """

    def __init__(self,
                 question,
                 true_branch,
                 false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


# In[74]:


def build_tree(rows):
    """Builds the tree.

    Rules of recursion: 1) Believe that it works. 2) Start by checking
    for the base case (no further information gain). 3) Prepare for
    giant stack traces.
    """

    # Try partitioing the dataset on each of the unique attribute,
    # calculate the information gain,
    # and return the question that produces the highest gain.
    gain, question = find_best_split(rows)

    # Base case: no further info gain
    # Since we can ask no further questions,
    # we'll return a leaf.
    if gain == 0:
        return Leaf(rows)

    # If we reach here, we have found a useful feature / value
    # to partition on.
    true_rows, false_rows = partition(rows, question)

    # Recursively build the true branch.
    true_branch = build_tree(true_rows)

    # Recursively build the false branch.
    false_branch = build_tree(false_rows)

    # Return a Question node.
    # This records the best feature / value to ask at this point,
    # as well as the branches to follow
    # dependingo on the answer.
    return Decision_Node(question, true_branch, false_branch)


# In[75]:


def print_tree(node, spacing=""):
    """World's most elegant tree printing function."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print (spacing + "Predict", node.predictions)
        return

    # Print the question at this node
    print (spacing + str(node.question))

    # Call this function recursively on the true branch
    print (spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")

    # Call this function recursively on the false branch
    print (spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")


# In[76]:


my_tree = build_tree(interesting_features.values)


# In[83]:


print_tree(my_tree)


# In[84]:


def classify(row, node):
    """See the 'rules of recursion' above."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        return node.predictions

    # Decide whether to follow the true-branch or the false-branch.
    # Compare the feature / value stored in the node,
    # to the example we're considering.
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)


# In[87]:


#######
# Demo:
# The tree predicts the 1st row of our
# training data is an apple with confidence 1.
print(interesting_features.iloc[250,:])
classify(interesting_features.iloc[250,:], my_tree)
#######


# In[92]:


def print_leaf(counts):
    """A nicer way to print the predictions at a leaf."""
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs


# In[93]:


#######
# Demo:
# Printing that a bit nicer
print_leaf(classify(interesting_features.iloc[0,:], my_tree))
#######


# In[96]:


#######
# Demo:
# On the second example, the confidence is lower
print_leaf(classify(interesting_features.iloc[0,:], my_tree))
#######

