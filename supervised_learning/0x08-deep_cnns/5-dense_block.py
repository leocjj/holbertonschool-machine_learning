#!/usr/bin/env python3
""" 0x08. Deep CNNs """
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Builds dense block as described in Densely Connected Convolutional Networks
    X is the output from the previous layer
    nb_filters is an integer representing the number of filters in X
    growth_rate is the growth rate for the dense block
    layers is the number of layers in the dense block
    Use the bottleneck layers used for DenseNet-B
    All weights use he normal initialization
    All convolutions are preceded by Batch Normalization and a rectified linear
        activation (ReLU), respectively
    Returns: The Xenated output of each layer within the Dense Block and
        the number of filters within the Xenated outputs, respectively
    """

    for layer in range(layers):
        Y = K.layers.BatchNormalization()(X)
        Y = K.layers.Activation('relu')(Y)
        Y = K.layers.Conv2D(growth_rate * 4, 1,
                                   kernel_initializer='he_normal')(Y)

        Y = K.layers.BatchNormalization()(Y)
        Y = K.layers.Activation('relu')(Y)
        Y = K.layers.Conv2D(growth_rate, 3, padding='same',
                                   kernel_initializer='he_normal')(Y)

        X = K.layers.concatenate([X, Y])

        nb_filters += growth_rate

    return X, nb_filters
