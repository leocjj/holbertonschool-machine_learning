#!/usr/bin/env python3
""" 0x06. Keras """

import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    Tests a neural network
    :param network: the network model to test
    :param data: the input data to test the model with
    :param labels: the correct one-hot labels of data
    :param verbose: boolean that determines if output should be printed during
        the testing process
    :return: loss and accuracy of the model with the testing data, respectively
    """

    return network.evaluate(x=data, y=labels, verbose=verbose)
