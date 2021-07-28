#!/usr/bin/env python3
"""
that represents a Multivariate Normal distribution
"""
import numpy as np


class MultiNormal:
    """Multinormal Class"""

    def __init__(self, data):
        """
        data is a numpy.ndarray of shape (d, n) containing the data set:
        n is the number of data points
        d is the number of dimensions in each data point
        If data is not a 2D numpy.ndarray, raise a TypeError with the message
        data must be a 2D numpy.ndarray
        If n is less than 2, raise a ValueError with the message data must
        contain multiple data points
        """
        if type(data) is not np.ndarray or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")

        mean = np.mean(data, axis=1).reshape((data.shape[0], 1))
        self.mean = mean
        self.cov = np.matmul(data - self.mean, data.T) / (data.shape[1] - 1)

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
