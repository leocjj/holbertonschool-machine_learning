#!/usr/bin/env python3
""" 0x01. Clustering """
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    initializes variables for a Gaussian Mixture Model:

    X is a numpy.ndarray of shape (n, d) containing the data set
    k is a positive integer containing the number of clusters
    Returns: pi, m, S, or None, None, None on failure
    pi is a numpy.ndarray of shape (k,) containing the priors for each cluster,
        initialized evenly
    m is a numpy.ndarray of shape (k, d) containing the centroid means for each
        cluster, initialized with K-means
    S is a numpy.ndarray of shape (k, d, d) containing the covariance matrices
        for each cluster, initialized as identity matrices
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2\
            or not isinstance(k, int) or k <= 0:
        return None, None, None

    d = X.shape[1]

    pi = np.full((k,), 1 / k)
    m, _ = kmeans(X, k)
    S = np.full((k, d, d), np.identity(d))

    return pi, m, S
