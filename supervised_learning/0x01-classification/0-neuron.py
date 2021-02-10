#!/usr/bin/env python3
""" 0x01. Classification """
import numpy as np


class Neuron:
    """ Class that defines a single neuron performing binary classification """

    def __init__(self, nx):
        """
        Defines a single neuron performing binary classification
        :param nx: number of input features to the neuron
        """
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.W = np.random.normal(size=nx).reshape(1, nx)
        self.b = 0
        self.A = 0
