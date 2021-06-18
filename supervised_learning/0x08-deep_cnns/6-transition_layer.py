#!/usr/bin/env python3
""" 0x08. Deep CNNs """
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    Builds transition layer as: Densely Connected Convolutional Networks
    X is the output from the previous layer
    nb_filters is an integer representing the number of filters in X
    compression is the compression factor for the transition layer
    Your code should implement compression as used in DenseNet-C
    All weights use he normal initialization
    All convolutions are preceded by Batch Normalization and a rectified linear
        activation (ReLU), respectively
    Returns: The output of the transition layer and the number of filters
        within the output, respectively
    """

    filters = nb_filters * compression

    out = K.layers.BatchNormalization()(X)
    out = K.layers.Activation('relu')(out)
    out = K.layers.Conv2D(int(filters), 1, kernel_initializer='he_normal')(out)
    out = K.layers.AvgPool2D(2)(out)

    return out, int(filters)
