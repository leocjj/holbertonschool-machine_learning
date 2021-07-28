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
        Calculates the PDF at a data point.
        Returns the value of the PDF.
        """
        if type(x) is not np.ndarray:
            raise TypeError("x must be a numpy.ndarray")

        d = self.cov.shape[0]

        if len(x.shape) != 2:
            raise ValueError('x must have the shape ({}, 1)'.format(d))

        if x.shape[1] != 1 or x.shape[0] != d:
            raise ValueError('x must have the shape ({}, 1)'.format(d))

        n = x.shape[0]

        mean = self.mean
        cov = self.cov
        cov_det = np.linalg.det(cov)
        cov_inv = np.linalg.inv(cov)

        # denominator
        den = np.sqrt(((2 * np.pi) ** n) * cov_det)

        # exponential term
        expo = -0.5 * np.matmul(np.matmul((x - mean).T, cov_inv), x - mean)

        PDF = (1 / den) * np.exp(expo[0][0])

        return PDF
