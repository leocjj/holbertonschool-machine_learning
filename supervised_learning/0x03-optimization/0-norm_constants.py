#!/usr/bin/env python3
""" 0x03. Optimization """
import numpy as np


def normalization_constants(X):
    """
    Calculates the normalization (standardization) constants of a matrix:
    :param X: is the numpy.ndarray of shape (m, nx) to normalize
        m is the number of data points
        nx is the number of features
    :return: mean and standard deviation of each feature.
    """

    mean = np.mean(X, axis=0)
    std_dev = np.std(X, axis=0)

    return mean, std_dev
