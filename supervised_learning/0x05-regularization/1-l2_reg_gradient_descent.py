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

    dZ = cache['A' + str(L)] - Y  # ERROR
    m1 = (1 / Y.shape[1])  # FACTOR TO DIVIDE BY NUMBER OF INPUT DATA
    for i in range(L, 0, -1):
        ''' BACKPROPAGATION USING GRADIENT DESCENT PLUS REGULARIZATION TERM'''
        cost_L2 = (lambtha * m1) * weights['W{}'.format(i)]
        dW = m1 * np.matmul(dZ, cache['A' + str(i - 1)].T) + cost_L2
        db = m1 * np.sum(dZ, axis=1, keepdims=True)
        weights['W' + str(i)] -= (alpha * dW)
        weights['b' + str(i)] -= (alpha * db)
        ''' dZ = (Wi * dZ).dA, dA is the activation function derivative.'''
        dZ = np.matmul(weights['W' + str(i)].T, dZ)
        ''' Apply activation function derivative (gradient). '''
        A = cache['A' + str(i - 1)]  # output of the activation function.
        dZ *= (1 - np.power(A, 2))  # f'(x) = 1 - f(x)^2
