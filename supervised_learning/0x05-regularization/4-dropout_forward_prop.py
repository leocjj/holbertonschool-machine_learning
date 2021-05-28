#!/usr/bin/env python3
""" 0x05. Regularization """

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout. All layers except the last
    should use the tanh activation function. The last layer should use the
    softmax activation function.
    :param X: numpy.ndarray of shape (nx, m) containing input data for network.
        nx is the number of input features
        m is the number of data points
    :param weights: dictionary of the weights and biases of the neural network.
    :param L: number of layers in the network.
    :param keep_prob: probability that a node will be kept.
    :return: a dictionary containing the outputs of each layer and the dropout
        mask used on each layer
    """

    cache = {'A0': X}
    for i in range(1, L + 1):
        z = np.matmul(weights["W" + str(i)], cache["A" + str(i - 1)])\
            + weights["b" + str(i)]
        dropout_matrix = np.random.binomial(1, keep_prob, size=z.shape)
        if i == L:
            e = np.exp(z)
            cache["A" + str(i)] = (e / np.sum(e, axis=0, keepdims=True))
        else:
            cache["A" + str(i)] = np.tanh(z) * dropout_matrix / keep_prob
            cache["D" + str(i)] = dropout_matrix

    return cache
