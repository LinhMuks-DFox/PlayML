import numpy as np


def accuracy_score(y_true: np.ndarray, y_predict: np.ndarray) -> float:
    assert y_true.shape[0] == y_predict.shape[0], \
        "the size of y_true must be equal to the size of y_predict"
    return np.sum(y_true == y_predict) / len(y_true)
