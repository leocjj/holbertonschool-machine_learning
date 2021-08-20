#!/usr/bin/env python3
"""
http://krasserm.github.io/2018/03/19/gaussian-processes/
Represents a noiseless 1D Gaussian process
"""
import numpy as np


class GaussianProcess:
    """
    Represents a noiseless 1D Gaussian process
    """
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Class constructor
        """
        # (t, 1) inputs already sampled with the black-box function
        self.X = X_init
        # (t, 1) outputs of the black-box function for each input in X
        self.Y = Y_init
        # length parameter for the kernel
        self.l = l
        # standard deviation given to the output of the black-box function
        self.sigma_f = sigma_f
        # Current covariance kernel matrix for the Gaussian process
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
        Calculates the covariance kernel matrix between two matrices
        """
        sqdist = \
            np.sum(X1**2, 1).reshape(-1, 1) \
            + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        return self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)
