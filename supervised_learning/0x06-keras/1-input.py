#!/usr/bin/env python3
""" 0x06. Keras """

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with the Keras library:
    :param nx: the number of input features to the network
    :param layers: list containing number of nodes in each layer of the network
    :param activations: list containing the activation functions used for each
        layer of the network
    :param lambtha: L2 regularization parameter
    :param keep_prob: probability that a node will be kept for dropout
    :return: keras model
    """

    inputs = K.Input(shape=(nx,))
    regularizer = K.regularizers.l2(lambtha)

    for i, layer in enumerate(layers):
        if i == 0:
            output = K.layers.Dense(layer, activation=activations[i],
                                    kernel_regularizer=regularizer)(inputs)
        else:
            dropout = K.layers.Dropout(1 - keep_prob)(output)
            output = K.layers.Dense(layer, activation=activations[i],
                                    kernel_regularizer=regularizer)(dropout)

    return K.Model(inputs=inputs, outputs=output)
