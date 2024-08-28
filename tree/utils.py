"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import pandas as pd
import numpy as np

def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Function to perform one hot encoding on the input data
    """
    return pd.get_dummies(X)
    pass

def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    return pd.api.types.is_float_dtype(y) or pd.api.types.is_integer_dtype(y)
    pass


def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    probabilities = Y.value_counts(normalize=True)
    return -np.sum(probabilities * np.log2(probabilities + 1e-9))
    pass


def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    probabilities = Y.value_counts(normalize=True)
    return 1 - np.sum(probabilities ** 2)
    pass


def information_gain(Y: pd.Series, attr: pd.Series, criterion: str) -> float:
    """
    Function to calculate the information gain using criterion (entropy, gini index or MSE)
    """
    if criterion == "information_gain":
        parent_entropy = entropy(Y)
        values = attr.unique()
        weighted_entropy = np.sum([
            len(Y[attr == value]) / len(Y) * entropy(Y[attr == value])
            for value in values
        ])
        return parent_entropy - weighted_entropy
    elif criterion == "gini_index":
        parent_gini = gini_index(Y)
        values = attr.unique()
        weighted_gini = np.sum([
            len(Y[attr == value]) / len(Y) * gini_index(Y[attr == value])
            for value in values
        ])
        return parent_gini - weighted_gini

    pass


def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon
    """

    # According to wheather the features are real or discrete valued and the criterion, find the attribute from the features series with the maximum information gain (entropy or varinace based on the type of output) or minimum gini index (discrete output).
    best_gain = -1
    best_split = None

    for feature in features:
        if check_ifreal(X[feature]):
            values = X[feature].unique()
            for value in values:
                gain = information_gain(y, X[feature] <= value, criterion)
                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature, value)
        else:
            values = X[feature].unique()
            gain = information_gain(y, X[feature], criterion)
            if gain > best_gain:
                best_gain = gain
                best_split = feature

    return best_split

    pass


def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    """
    Funtion to split the data according to an attribute.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    attribute: attribute/feature to split upon
    value: value of that attribute to split upon

    return: splitted data(Input and output)
    """

    # Split the data based on a particular value of a particular attribute. You may use masking as a tool to split the data.
    if isinstance(attribute, tuple):  # Real value split
        feature, split_value = attribute
        left_mask = X[feature] <= split_value
        right_mask = X[feature] > split_value
    else:  # Discrete value split
        feature = attribute
        left_mask = X[feature] == value
        right_mask = X[feature] != value

    return X[left_mask], y[left_mask], X[right_mask], y[right_mask]
    pass

