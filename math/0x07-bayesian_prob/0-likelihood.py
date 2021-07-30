#!/usr/bin/env python3
"""The likelihood, bayesian"""
import numpy as np


def likelihood(x, n, P):
    """
    calculates the likelihood of obtaining this data given various hypothetical
     probabilities of developing severe side effects:

    x is the number of patients that develop severe side effects
    n is the total number of patients observed
    P is a 1D numpy.ndarray containing the various hypothetical probabilities
        of developing severe side effects
    If n is not a positive integer, raise a ValueError with the message n must
        be a positive integer
    If x is not an integer that is greater than or equal to 0, raise a
        ValueError with the message x must be an integer that is greater than
         or equal to 0
    If x is greater than n, raise a ValueError with the message x cannot be
        greater than n
    If P is not a 1D numpy.ndarray, raise a TypeError with the message P must
        be a 1D numpy.ndarray
    If any value in P is not in the range [0, 1], raise a ValueError with the
        message All values in P must be in the range [0, 1]
    Returns: a 1D numpy.ndarray containing the likelihood of obtaining the
        data, x and n, for each probability in P, respec
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) and x < 0:
        raise ValueError("x must be an integer that is",
                         " greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if np.any(P > 1) or np.any(P < 0):
        raise ValueError("All values in P must be in the range [0, 1]")
    likelihoods = ((np.math.factorial(n)) / (np.math.factorial(x) *
                   (np.math.factorial(n - x))) * P ** x * (1 - P) ** (n - x))

    return likelihoods
