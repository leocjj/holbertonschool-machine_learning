#!/usr/bin/env python3
""" 0x01. Classification """
import numpy as np


def one_hot_encode(Y, classes):
    """
    Function that converts a numeric label vector into a one-hot matrix.
    :param Y: is a np.ndarray with shape (m,) containing numeric class labels.
    :param classes: is the maximum number of classes found in Y
    :return: a one-hot encoding of Y with shape (classes, m), or None on failure
    """
    A = np.zeros((classes, len(Y)))
    for i in range(classes):
        A[Y[i]][i] = 1
    return A
