#!/usr/bin/env python3
""" 0x03. Optimization """
import numpy as np


def normalize(X, m, s):
    """
    Normalizes (standardizes) a matrix:
    :param X: is the numpy.ndarray of shape (m, nx) to normalize
        m is the number of data points
        nx is the number of features
    :param m: ndarray of shape (nx,) contains the mean of all features of X
    :param s: ndarray of shape (nx,) contains the standard deviation of all
        features of X
    :return: normalized X matrix
    """

    return (X - m) / s
