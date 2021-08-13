#!/usr/bin/env python3
""" 0x02. Hidden Markov Models """
import numpy as np


def regular(P):
    """
    determines the steady state probabilities of a regular markov chain
    """
    if len(P.shape) != 2 or P.shape[0] != P.shape[1] or P.shape[0] < 1:
        return None

    P = np.linalg.matrix_power(P, 100)
    if np.any(P <= 0):
        return None
    ss_prob = np.array([P[0]])

    return ss_prob
