from typing import Union
import pandas as pd


def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """

    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert y_hat.size == y.size
    # TODO: Write here
    correct_predictions = (y_hat == y).sum()
    return correct_predictions / y.size
    pass


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    assert y_hat.size == y.size, "The size of y_hat and y must be the same."
    
    # Convert cls to integer if it's a string
    if isinstance(cls, str):
        cls = int(cls)

    true_positive = ((y_hat == cls) & (y == cls)).sum()
    predicted_positive = (y_hat == cls).sum()

    if predicted_positive == 0:
        return 0.0
    
    return true_positive / predicted_positive
    pass


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    assert y_hat.size == y.size, "The size of y_hat and y must be the same."
    
    # Convert cls to integer if it's a string
    if isinstance(cls, str):
        cls = int(cls)

    true_positive = ((y_hat == cls) & (y == cls)).sum()
    actual_positive = (y == cls).sum()

    if actual_positive == 0:
        return 0.0
    
    return true_positive / actual_positive
    pass


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """
    assert y_hat.size == y.size, "The size of y_hat and y must be the same."
    mse = ((y_hat - y) ** 2).mean()
    return (mse)**(1/2)
    pass


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    assert y_hat.size == y.size, "The size of y_hat and y must be the same."
    return (y_hat - y).abs().mean()
    pass
