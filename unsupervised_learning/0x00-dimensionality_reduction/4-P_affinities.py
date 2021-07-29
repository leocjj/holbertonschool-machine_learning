#!/usr/bin/env python3
"""
Calculates the symmetric P affinities of a data set
"""


import numpy as np
P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """
    Returns: P, a numpy.ndarray of shape (n, n)
    containing the symmetric P affinities
    """
    return None
