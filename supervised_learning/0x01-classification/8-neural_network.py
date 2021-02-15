#!/usr/bin/env python3
""" 0x01. Classification """
import numpy as np


class NeuralNetwork:
    """ Class that defines a neural network with one hidden layer performing
        binary classification: """

    def __init__(self, nx, nodes):
        """
        Defines a single layer performing binary classification
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
        self.W1 = np.random.standard_normal((nodes, nx))
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0
        # Output neuron
        self.W2 = np.random.standard_normal((1, nodes))
        self.b2 = 0
        self.A2 = 0
