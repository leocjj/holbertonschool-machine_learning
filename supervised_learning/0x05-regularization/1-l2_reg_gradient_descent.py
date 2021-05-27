#!/usr/bin/env python3
""" 0x05. Regularization """

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    updates the weights and biases of a neural network using gradient descent
    with L2 regularization. The neural network uses tanh activations on each
    layer except the last, which uses a softmax activation.
    :param Y: Y is a one-hot numpy.ndarray of shape (classes, m) that contains
        the correct labels for the data classes is the number of classes.
        m is the number of data points
    :param weights: dictionary of the weights and biases of the neural network
    :param cache: dictionary of the outputs of each layer of the neural network
    :param alpha: learning rate
    :param lambtha: L2 regularization parameter
    :param L:  number of layers of the network
    :return: None. Weights and biases of the network should be updated in place
    """

    # ERROR
    dz = cache['A' + str(L)] - Y
    # FACTOR TO DIVIDE BY NUMBER OF INPUT DATA
    m = (Y.shape[1])
    for i in range(L, 0, -1):
        cost_L2 = (lambtha / m) * weights['W'+str(i)]
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
        dW = ((1 / m) * np.matmul(dz, cache['A'+str(i-1)].T)) + cost_L2
        dz = np.matmul(weights['W'+str(i)].T, dz) *\
            ((1 - cache['A'+str(i-1)] ** 2))
        weights['W'+str(i)] = weights['W'+str(i)] -\
            (alpha * dW)
        weights['b'+str(i)] = weights['b'+str(i)] -\
            (alpha * db)
