# !/usr/bin/env python3
""" 0x06. Multivariate Probability """
import numpy as np


def mean_cov(X):
    """
    Calculates the mean and covariance of a data set:
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    n, d = X.shape
    if n < 2:
        raise ValueError("X must contain multiple data points")

    mean = np.mean(X, axis=0)
    dev = X - mean
    cov = np.matmul(dev.T, dev) / (n - 1)

    return mean, cov
