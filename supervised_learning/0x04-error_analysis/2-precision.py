#!/usr/bin/env python3
""" 0x04. Error Analysis """

import numpy as np


def precision(confusion):
    """
    Function that calculates the precision for each class in a confusion matrix
    :param confusion: a confusion numpy.ndarray of shape (classes, classes)
        where row indices represent the correct labels and column indices
        represent the predicted labels
        classes is the number of classes
    :return: numpy.ndarray of shape (classes,) containing the precision of
        each class
    """

    return np.diagonal(confusion) / np.sum(confusion, axis=0)
