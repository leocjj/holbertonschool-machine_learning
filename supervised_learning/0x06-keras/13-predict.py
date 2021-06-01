#!/usr/bin/env python3
""" 0x06. Keras """

import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    Makes a prediction using a neural network
    :param network: the network model to make the prediction with
    :param data: the input data to make the prediction with
    :param verbose: boolean that determines if output should be printed during
        the prediction process
    :return: prediction for the data
    """

    return network.predict(x=data, verbose=verbose)
