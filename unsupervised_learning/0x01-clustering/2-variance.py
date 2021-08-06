#!/usr/bin/env python3
""" 0x01. Clustering """
import numpy as np


def variance(X, C):
    """
    calculates the total intra-cluster variance for a data set:

    X is a numpy.ndarray of shape (n, d) containing the data set
    C is a numpy.ndarray of shape (k, d) containing the centroid means
        for each cluster
    Returns: var, or None on failure
        var is the total variance
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2\
            or not isinstance(C, np.ndarray) or len(C.shape) != 2\
            or X.shape[0] < C.shape[0] or X.shape[1] != C.shape[1]:
        return None

    d = np.linalg.norm(X[:, np.newaxis] - C, axis=-1)
    m = np.min(d, axis=-1)

    return np.sum(np.sum(m ** 2))
