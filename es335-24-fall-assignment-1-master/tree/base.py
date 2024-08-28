"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *

np.random.seed(42)


@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """
        self.tree = self._fit(X, y, depth=0)        # If you wish your code can have cases for different types of input and output data (discrete, real)
        # Use the functions from utils.py to find the optimal attribute to split upon and then construct the tree accordingly.
        # You may(according to your implemetation) need to call functions recursively to construct the tree. 

        pass
    
    def _fit(self, X: pd.DataFrame, y: pd.Series, depth: int) -> Any:
        """
        Recursive function to fit the tree.
        """
        if depth >= self.max_depth or len(y.unique()) == 1:
            return y.mode()[0]

        features = X.columns
        best_split = opt_split_attribute(X, y, self.criterion, features)

        if best_split is None:
            return y.mode()[0]

        if isinstance(best_split, tuple):
            feature, value = best_split
            left_mask = X[feature] <= value
            right_mask = X[feature] > value
        else:
            feature = best_split
            left_mask = X[feature] == value
            right_mask = X[feature] != value

        left_tree = self._fit(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._fit(X[right_mask], y[right_mask], depth + 1)

        return (feature, value, left_tree, right_tree) if isinstance(best_split, tuple) else (feature, left_tree, right_tree)
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """

        # Traverse the tree you constructed to return the predicted values for the given test inputs.
        return X.apply(lambda row: self._predict_row(row, self.tree), axis=1)
        pass
    
    def _predict_row(self, row: pd.Series, tree: Any) -> Any:
        """
        Predict the class label for a single row.
        """
        if not isinstance(tree, tuple):
            return tree

        if len(tree) == 3:
            feature, value, left_tree, right_tree = tree
            if row[feature] <= value:
                return self._predict_row(row, left_tree)
            else:
                return self._predict_row(row, right_tree)
        else:
            feature, left_tree, right_tree = tree
            if row[feature] in left_tree:
                return self._predict_row(row, left_tree)
            else:
                return self._predict_row(row, right_tree)
    
    def plot(self) -> None:
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        self._plot_node(self.tree, depth=0, prefix="")
        plt.show()
        pass
    def _plot_node(self, node: Any, depth: int, prefix: str) -> None:
        """
        Recursive function to plot the nodes.
        """
        if not isinstance(node, tuple):
            print(f"{prefix}Predict: {node}")
            return

        feature, value, left_tree, right_tree = node
        print(f"{prefix}Feature {feature} <= {value}?")
        print(f"{prefix}True:")
        self._plot_node(left_tree, depth + 1, prefix + "  ")
        print(f"{prefix}False:")
        self._plot_node(right_tree, depth + 1, prefix + "  ")
