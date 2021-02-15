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

        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        for i in range(self.L):
            if layers[i] < 1 or type(layers[i]) is not int:
                raise TypeError("layers must be a list of positive integers")

            w_i = "W" + str(i + 1)
            b_i = "b" + str(i + 1)

            if i == 0:
                self.weights[w_i] = np.random.randn(layers[i], nx)\
                                            * np.sqrt(2 / nx)
            if i > 0:
                self.weights[w_i] = np.random.randn(layers[i], layers[i - 1])\
                    * np.sqrt(2 / layers[i - 1])
            self.weights[b_i] = np.zeros((layers[i], 1))
