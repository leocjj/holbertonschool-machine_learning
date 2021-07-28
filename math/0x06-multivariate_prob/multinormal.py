#!/usr/bin/env python3
"""
Represents a Multivariate Normal distribution
"""


import numpy as np


class MultiNormal:
    """
    Represents a Multivariate Normal distribution
    """
    def __init__(self, data):
        """
        Class constructor
        """
        if type(data) is not np.ndarray or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape

        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1).reshape(d, 1)

        deviation = data - self.mean

        self.cov = np.matmul(deviation, deviation.T) / (n - 1)


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
