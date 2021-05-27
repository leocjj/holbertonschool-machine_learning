#!/usr/bin/env python3
""" 0x05. Regularization """

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """

    :param cost: cost of the network without L2 regularization
    :param lambtha: the regularization parameter
    :param weights: dictionary of the weights and biases (numpy.ndarrays) of
        the neural network
    :param L: number of layers in the neural network.
    :param m: number of data points used.
    :return: cost of the network accounting for L2 regularization
        cost = cost + (lambda/2m).sum(norm(Wi)^2)
    """

    l2_norms_power2 = [np.linalg.norm(weights['W{}'.format(i)]) ** 2
                       for i in range(1, L + 1)]
    return cost + (lambtha / (2 * m)) * sum(l2_norms_power2)
