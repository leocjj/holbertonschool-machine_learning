#!/usr/bin/env python3
""" 0x06. Keras """

import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    Converts a label vector into a one-hot matrix.
    The last dimension of the one-hot matrix must be the number of classes.
    :param labels:
    :param classes:
    :return: one-hot matrix
    """

    return K.utils.to_categorical(labels, num_classes=classes)
