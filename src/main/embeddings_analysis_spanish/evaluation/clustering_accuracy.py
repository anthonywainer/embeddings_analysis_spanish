import numpy as np
from numpy import int64, zeros

from embeddings_analysis_spanish.utils.assigment import linear_assignment


def acc(y_true: np.array, y_predicted: np.array) -> np.array:
    """
    Calculate clustering accuracy
    :param y_true: labels with shape `(n_samples,)`
    :param y_predicted: predicted labels with shape `(n_samples,)`
    :return: accuracy, in [0,1]
    """
    y_true = y_true.astype(int64)
    assert y_predicted.size == y_true.size

    max_value = max(y_predicted.max(), y_true.max()) + 1
    weights = zeros((max_value, max_value), dtype=int64)

    for i in range(y_predicted.size):
        weights[y_predicted[i], y_true[i]] += 1

    assignments = linear_assignment(weights.max() - weights)
    return sum([weights[i, j] for i, j in assignments]) * 1.0 / y_predicted.size
