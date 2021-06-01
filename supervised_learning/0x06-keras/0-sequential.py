#!/usr/bin/env python3
""" 0x06. Keras """

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """

    :param nx: number of input features to the network
    :param layers: list containing number of nodes in each layer of the network
    :param activations: list containing the activation functions used for each
        layer of the network
    :param lambtha: L2 regularization parameter
    :param keep_prob: probability that a node will be kept for dropout
    :return: keras model
    """

    model = K.Sequential()
    regularizer = K.regularizers.l2(lambtha)

    for i, layer in enumerate(layers):
        if i == 0:
            model.add(K.layers.Dense(layer, activation=activations[i],
                                     kernel_regularizer=regularizer,
                                     input_shape=(nx,)))
        else:
            model.add(K.layers.Dense(layer, activation=activations[i],
                                     kernel_regularizer=regularizer))
        if i < len(layers) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))

    return model
