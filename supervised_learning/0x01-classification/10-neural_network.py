#!/usr/bin/env python3
""" 0x01. Classification """
import numpy as np


def sigmoid(x):
    """
    https://stackoverflow.com/questions/3985619/
    how-to-calculate-a-logistic-sigmoid-function-in-python
    Y = 1. / (1. + np.exp(-x))
    :param x: int or array. Use math.ext instead of np.exp for integers.
    :return: sigmoid function of x
    """
    return np.exp(-np.logaddexp(0., -x))


class NeuralNetwork:
    """ Class that defines a neural network with one hidden layer performing
        binary classification: """

    def __init__(self, nx, nodes):
        """
        Defines a single neuron performing binary classification
        :param nx: number of input features to the neuron
        :param nodes: is the number of nodes found in the hidden layer
        """
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if not isinstance(nodes, int):
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')
        # Hidden layer
        self.__W1 = np.random.standard_normal((nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        # Output neuron
        self.__W2 = np.random.standard_normal((1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """  getter method """
        return self.__W1

    @property
    def b1(self):
        """  getter method """
        return self.__b1

    @property
    def A1(self):
        """  getter method """
        return self.__A1

    @property
    def W2(self):
        """  getter method """
        return self.__W2

    @property
    def b2(self):
        """  getter method """
        return self.__b2

    @property
    def A2(self):
        """  getter method """
        return self.__A2

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron,
            using a sigmoid activation function.
        :param X: is a numpy.ndarray with shape (nx, m) that contains the input
            nx is the number of input features to the neuron,
            m is the number of examples.
        :return: the private attribute __A
        """
        self.__A1 = sigmoid(np.matmul(self.W1, X) + self.b1)
        self.__A2 = sigmoid(np.matmul(self.W2, self.__A1) + self.b2)
        return self.A1, self.A2
