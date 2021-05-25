#!/usr/bin/env python3
""" 0x04. Error Analysis """

import numpy as np
sensitive = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Function that calculates the F1 score of a confusion matrix
    :param confusion: a confusion numpy.ndarray of shape (classes, classes)
        where row indices represent the correct labels and column indices
        represent the predicted labels
        classes is the number of classes
    :return: numpy.ndarray of shape (classes,) containing the F1 score of
        each class
    TP = np.diag(confusion)
    PP = np.sum(confusion, axis=0)
    P = np.sum(confusion, axis=1)
    # TP / (TP + 0.5(FP + FN))
    return 2 * TP / (PP + P)
    """

    return 2 / (pow(precision(confusion), -1) + pow(sensitive(confusion), -1))
