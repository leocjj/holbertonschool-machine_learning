#!/usr/bin/env python3
""" 0x02. Tensorflow """
import tensorflow as tf


def create_layer(prev, n, activation):
    """
    Function to create a layer
    :param prev: is the tensor output of the previous layer
    :param n: is the number of nodes in the layer to create
    :param activation: is the activation function that the layer should use
    :return: the tensor output of the layer
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=activation, name='layer',
                            kernel_initializer=init)
    return layer(prev)
