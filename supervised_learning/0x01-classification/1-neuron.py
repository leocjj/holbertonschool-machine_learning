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
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """  getter method """
        return self.__W

    @property
    def b(self):
        """  getter method """
        return self.__b

    @property
    def A(self):
        """  getter method """
        return self.__A
