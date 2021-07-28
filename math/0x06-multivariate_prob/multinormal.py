#!/usr/bin/env python3
""" 0x06. Multivariate Probability """
import numpy as np


class MultiNormal():
    """
    represents a Multivariate Normal distribution:
    Set the public instance variables:
    mean - a numpy.ndarray of shape (d, 1) containing the mean of data
    cov - a numpy.ndarray of shape (d, d) containing the covariance matrix data
    """

    def __init__(self, data):
        """
        data is a numpy.ndarray of shape (d, n) containing the data set:
        n is the number of data points
        d is the number of dimensions in each data point
        """
        if type(data) is not np.ndarray or len(data.shape) != 2:
            raise TypeError('data must be a 2D numpy.ndarray')
        if data.shape[1] < 2:
            raise ValueError('data must contain multiple data points')
        d, n = data.shape
        self.mean = data.mean(axis=1, keepdims=True)
        self.dev = data - self.mean
        self.cov = np.matmul(self.dev, self.dev.T) / (n - 1)

    def pdf(self, x):
        """
        x is a numpy.ndarray of shape (d, 1) containing the data point whose
            PDF should be calculated
        d is the number of dimensions of the Multinomial instance
        If x is not a numpy.ndarray, raise a TypeError with the message x must
            be a numpy.ndarray
        If x is not of shape (d, 1), raise a ValueError with the message x must
            have the shape ({d}, 1)
        Returns the value of the PDF
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        d = self.mean.shape[0]
        if len(x.shape) != 2 or x.shape != (d, 1):
            raise ValueError("x must have the shape ({}, 1)".format(d))

        res = np.exp(np.matmul(np.matmul((x - self.mean).T,
                                         np.linalg.inv(self.cov)),
                               (x - self.mean)
                               ) / -2)\
            / np.sqrt(pow(2 * np.pi, x.shape[0])
                      * np.linalg.det(self.cov)
                      )

        return res[0][0]
