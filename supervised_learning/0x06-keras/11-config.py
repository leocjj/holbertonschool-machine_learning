#!/usr/bin/env python3
""" 0x06. Keras """

import tensorflow.keras as K


def save_config(network, filename):
    """
    Saves a model’s configuration in JSON format
    :param network: model whose configuration should be saved
    :param filename: path of the file that the configuration should be saved to
    :return: None
    """

    json_string = network.to_json()

    with open(filename, 'w') as f:
        f.write(json_string)

    return None


def load_config(filename):
    """
    Loads a model with a specific configuration:
    :param filename: path of file containing the model’s configuration in JSON
    :return: None
    """

    with open(filename, "r") as f:
        network_string = f.read()

    return K.models.model_from_json(network_string)
