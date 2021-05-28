#!/usr/bin/env python3
""" 0x05. Regularization """

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network with Dropout regularization using
    gradient descent. All layers use thetanh activation function except the
    last, which uses the softmax activation function.
    :param Y: one-hot numpy.ndarray of shape (classes, m) that contains the
        correct labels for the data:
        classes is the number of classes
        m is the number of data points
    :param weights: dictionary of the weights and biases of the neural network.
    :param cache: dictionary of the outputs and dropout masks of each layer of
        the neural network.
    :param alpha: the learning rate
    :param keep_prob: the probability that a node will be kept
    :param L: the number of layers of the network
    :return: None. The weights of the network should be updated in place.
    """

    # ERROR
    dZ = cache['A' + str(L)] - Y
    # FACTOR TO DIVIDE BY NUMBER OF INPUT DATA
    m1 = (1 / Y.shape[1])
    for i in range(L, 0, -1):
        ''' BACKPROPAGATION USING GRADIENT DESCENT'''
        dW = (m1 * np.matmul(dZ, cache['A' + str(i - 1)].T))
        db = m1 * np.sum(dZ, axis=1, keepdims=True)

        ''' dZ = (Wi * dZ).dA, dA is the activation function derivative.'''
        dZ = np.matmul(weights['W' + str(i)].T, dZ)
        ''' Apply activation function derivative (gradient). '''
        # output of the activation function.
        A = cache['A' + str(i - 1)]
        # f'(x) = 1 - f(x)^2
        if i > 1:
            dZ *= (1 - np.power(A, 2)) * (cache['D' + str(i - 1)] / keep_prob)

        weights['W' + str(i)] -= (alpha * dW)
        weights['b' + str(i)] -= (alpha * db)
