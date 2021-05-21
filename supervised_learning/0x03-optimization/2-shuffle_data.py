#!/usr/bin/env python3
""" 0x03. Optimization """
import numpy as np


def shuffle_data(X, Y):
    """
    shuffles the data points in two matrices the same way:
    :param X: is the numpy.ndarray of shape (m, nx) to normalize
        m is the number of data points
        nx is the number of features
    :param Y: is the second numpy.ndarray of shape (m, ny) to shuffle
        m is the same number of data points as in X
        ny is the number of features in Y
    :return: shuffled X and Y matrices
    """

    shuffler = np.random.permutation(len(X))

    return X[shuffler], Y[shuffler]
