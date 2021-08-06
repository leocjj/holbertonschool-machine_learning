#!/usr/bin/env python3
""" 0x01. Clustering """
import numpy as np


def initialize(X, k):
    """
    initializes cluster centroids for K-means:

    X is a numpy.ndarray of shape (n, d) containing the dataset that will
        be used for K-means clustering
    n is the number of data points
    d is the number of dimensions for each data point
    k is a positive integer containing the number of clusters
    The cluster centroids should be initialized with a multivariate uniform
        distribution along each dimension in d:
    The minimum values for the distribution should be the minimum values
        of X along each dimension in d
    The maximum values for the distribution should be the maximum values
        of X along each dimension in d
    Returns: a numpy.ndarray of shape (k, d) containing the initialized
        centroids for each cluster, or None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2\
            or not isinstance(k, int) or k <= 0 or k >= X.shape[0]:
        return None

    return np.random.uniform(np.min(X, axis=0), np.max(X, axis=0),
                             (k, X.shape[1]))


def kmeans(X, k, iterations=1000):
    """
    that performs K-means on a dataset:

    X is a numpy.ndarray of shape (n, d) containing the dataset
    n is the number of data points
    d is the number of dimensions for each data point
    k is a positive integer containing the number of clusters
    iterations is a positive integer containing the maximum number of
        iterations that should be performed
    If no change in the cluster centroids occurs between iterations, your
        function should return
    Initialize the cluster centroids using a multivariate uniform distribution
        (based on 0-initialize.py)
    If a cluster contains no data points during the update step, reinitialize
        its centroid
    Returns: C, clss, or None, None on failure
        C is a numpy.ndarray of shape (k, d) containing the centroid means for
            each cluster
        clss is a numpy.ndarray of shape (n,) containing the index of the
            cluster in C that each data point belongs to
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2\
            or not isinstance(k, int) or k <= 0 or k >= X.shape[0]:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    _min = np.min(X, axis=0)
    _max = np.max(X, axis=0)
    C1 = np.random.uniform(_min, _max, size=(k, X.shape[1]))

    for i in range(iterations):
        clss = np.argmin(np.linalg.norm(X[:, None] - C1, axis=-1), axis=-1)
        C2 = np.copy(C1)
        for c in range(k):
            if c not in clss:
                C2[c] = np.random.uniform(_min, _max)
            else:
                C2[c] = np.mean(X[clss == c], axis=0)
        if np.array_equal(C2, C1):
            return (C1, clss)
        else:
            C1 = C2

    clss = np.argmin(np.linalg.norm(X[:, None] - C1, axis=-1), axis=-1)

    return C1, clss
