#!/usr/bin/env python3
""" 0x03. Optimization """
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes an unactivated output of a neural network using batch
        normalization.
    :param Z: ndarray of shape (m, n) that should be normalized
        m is the number of data points
        n is the number of features in Z
    :param gamma: ndarray of shape (1, n) containing the scales used for batch
        normalization
    :param beta: ndarray of shape (1, n) containing the offsets used for batch
        normalization
    :param epsilon: a small number used to avoid division by zero
    :return: normalized Z matrix
    """

    mean = np.sum(Z, axis=0) / Z.shape[0]
    variance = np.sum(np.power(Z - mean, 2), axis=0) / Z.shape[0]
    Z_normalized = (Z - mean) / np.sqrt(variance + epsilon)
    Z_scaled = gamma * Z_normalized + beta

    return Z_scaled
