#!/usr/bin/env python3
""" 0x02. Hidden Markov Models """
import numpy as np


def absorbing(P):
    """
    determines if a markov chain is absorbing
    """
    if not isinstance(P, np.ndarray)\
            or len(P.shape) != 2\
            or P.shape[0] != P.shape[1]\
            or P.shape[0] < 1:
        return None

    if np.all(np.diag(P) == 1):
        return True

    if P[0, 0] != 1:
        return False

    P = P[1:, 1:]

    if np.all(np.count_nonzero(P, axis=0) > 2):
        return True
    else:
        return False

