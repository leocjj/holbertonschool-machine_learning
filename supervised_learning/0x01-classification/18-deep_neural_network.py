#!/usr/bin/env python3
""" 0x01. Classification """
import numpy as np


class DeepNeuralNetwork:
    """ Class that defines a deep neural network performing binary
        classification: """

    def __init__(self, nx, layers):
        """
        Defines a neural network performing binary classification
        :param nx: number of input features to the neuron
        :param layers: is the number of nodes found in the hidden layer
        """
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if not isinstance(layers, list) or not list:
            raise TypeError('layers must be a list of positive integers')
        if not np.issubdtype(np.array(layers).dtype, np.integer) or\
                not all(np.array(layers) >= 1):
            raise TypeError('layers must be a list of positive integers')

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {'W1':
                          np.random.randn(layers[0], nx) * np.sqrt(2 / nx),
                          'b1': np.zeros((layers[0], 1))
                          }

        for i in range(1, self.__L):
            self.__weights["W" + str(i + 1)] =\
                np.random.randn(layers[i], layers[i - 1]) *\
                np.sqrt(2 / layers[i - 1])
            self.__weights["b" + str(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """  getter method """
        return self.__L

    @property
    def cache(self):
        """  getter method """
        return self.__cache

    @property
    def weights(self):
        """  getter method """
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron,
            using a sigmoid activation function.
        :param X: is a numpy.ndarray with shape (nx, m) that contains the input
            nx is the number of input features to the neuron,
            m is the number of examples.
        :return: the private attributes __A1, __A2
        """

        self.__cache["A0"] = X
        for i in range(self.__L):
            w_i = "W" + str(i + 1)
            b_i = "b" + str(i + 1)
            Za = np.matmul(self.__weights[w_i], self.__cache["A" + str(i)])
            Z = Za + self.__weights[b_i]
            self.__cache["A" + str(i + 1)] = 1 / (1 + np.exp(-Z))
        return self.__cache["A" + str(self.__L)], self.__cache
