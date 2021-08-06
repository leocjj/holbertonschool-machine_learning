#!/usr/bin/env python3
""" 0x01. Clustering """
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    calculates the expectation step in the EM algorithm for a GMM:

    X is a numpy.ndarray of shape (n, d) containing the data set
    pi is a numpy.ndarray of shape (k,) containing the priors for each cluster
    m is a numpy.ndarray of shape (k, d) containing the centroid means for
        each cluster
    S is a numpy.ndarray of shape (k, d, d) containing the covariance matrices
        for each cluster
    Returns: g, l, or None, None on failure
        g is a numpy.ndarray of shape (k, n) containing the posterior
            probabilities for each data point in each cluster
        l is the total log likelihood
    """
    if not isinstance(X, np.ndarray) or not isinstance(m, np.ndarray)\
            or not isinstance(S, np.ndarray) or not isinstance(pi, np.ndarray):
        return None, None

    if len(X.shape) != 2 or len(S.shape) != 3\
            or len(pi.shape) != 1 or len(m.shape) != 2\
            or m.shape[1] != X.shape[1]\
            or S.shape[2] != S.shape[1]\
            or S.shape[0] != pi.shape[0]\
            or S.shape[0] != m.shape[0]\
            or np.min(pi) < 0:
        return None, None

    k = pi.shape[0]
    g = np.zeros([k, X.shape[0]])
    for i in range(k):
        P = pdf(X, m[i], S[i])
        g[i] = pi[i] * P
    likelyhood = np.sum(np.log(g.sum(axis=0)))
    g = g / g.sum(axis=0)

    return g, likelyhood
