#!/usr/bin/env python3
""" 0x04. Error Analysis """

import numpy as np


def specificity(confusion):
    """
    Function that calculates specificity for each class in a confusion matrix
    :param confusion: a confusion numpy.ndarray of shape (classes, classes)
        where row indices represent the correct labels and column indices
        represent the predicted labels
        classes is the number of classes
    :return: numpy.ndarray of shape (classes,) containing the specificity of
        each class
    """

    ALL = np.sum(confusion)
    TP = np.diag(confusion)
    PP = np.sum(confusion, axis=0)
    P = np.sum(confusion, axis=1)

    return (ALL - PP - P + TP) / (ALL - P)
