#!/usr/bin/env python3
"""
0x03. Hyperparameter Tuning
"""
import numpy as np


class GaussianProcess:
    """
    class that represents a noiseless 1D Gaussian process
    """
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        init of gp
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """
        calculates the covariance kernel matrix between two matrices
        """
        cov = \
            np.sum(X1**2, 1).reshape(-1, 1) \
            + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        return self.sigma_f ** 2 * np.exp(-0.5 / self.l ** 2 * cov)

    def predict(self, X_s):
        """ Predicts the mean and standard deviation of points in a
        Gaussian process """
        K = self.kernel(self.X, self.X)
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(K)
        mu_s = K_s.T.dot(K_inv).dot(self.Y)
        sigma = K_ss - K_s.T.dot(K_inv).dot(K_s)

        return mu_s.flatten(), np.diag(sigma)
