#!/usr/bin/env python3
"""
Performs a t-SNE transformation
"""


import numpy as np
pca = __import__('1-pca').pca
P_affinities = __import__('4-P_affinities').P_affinities
grads = __import__('6-grads').grads
cost = __import__('7-cost').cost


def tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500):
    """
    Returns: Y, a numpy.ndarray of shape (n, ndim)
    containing the optimized low dimensional transformation of X
    """
    return None
