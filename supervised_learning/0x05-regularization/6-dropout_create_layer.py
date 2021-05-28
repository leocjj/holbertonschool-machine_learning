#!/usr/bin/env python3
""" 0x05. Regularization """

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    Creates a layer of a neural network using dropout
    :param prev: tensor containing the output of the previous layer
    :param n: number of nodes the new layer should contain
    :param activation: activation function that should be used on the layer
    :param keep_prob: probability that a node will be kept
    :return: the output of the new layer
    """

    regularizer = tf.layers.Dropout(keep_prob)
    weights = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    outputs = tf.layers.Dense(units=n,
                              activation=activation,
                              kernel_initializer=weights,
                              kernel_regularizer=regularizer)
    return outputs(prev)
