#!/usr/bin/env python3
""" 0x06. Keras """

import tensorflow.keras as K


def save_model(network, filename):
    """
    Saves an entire model
    :param network: the model to save
    :param filename: the path of the file that the model should be saved to
    :return: None
    """

    network.save(filename)
    return None


def load_model(filename):
    """
    Loads an entire model
    :param filename: the path of the file that the model should be loaded from
    :return: the loaded model
    """

    return K.models.load_model(filename)
