#!/usr/bin/env python3
""" 0x06. Keras """

import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """
    Saves a model’s weights
    :param network: the model whose weights should be saved
    :param filename: the path of the file that the weights should be saved to
    :param save_format: the format in which the weights should be saved
    :return: None
    """

    network.save_weights(filename, save_format=save_format)
    return None


def load_weights(network, filename):
    """
    Loads a model’s weights
    :param network: model to which the weights should be loaded
    :param filename: path of the file that the weights should be loaded from
    :return: None
    """

    network.load_weights(filename)
    return None
