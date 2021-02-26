#!/usr/bin/env python3
""" 0x03. Optimization """
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in tensorflow:
        Your layer should incorporate two trainable parameters, gamma and beta,
        initialized as vectors of 1 and 0 respectively. You should use an
        epsilon of 1e-8
    :param prev: activated output of the previous layer
    :param n: number of nodes in the layer to be created
    :param activation: activation function that should be used on the output
        of the layer
    :return: tensor of the activated output for the layer
    """

    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    y = tf.layers.Dense(units=n, kernel_initializer=init, name='layer')
    x = y(prev)

    mean, variance = tf.nn.moments(x, axes=[0])
    gamma = tf.Variable(tf.constant(1.0, shape=[n]), trainable=True)
    beta = tf.Variable(tf.constant(0.0, shape=[n]), trainable=True)
    epsilon = 1e-8

    norma = tf.nn.batch_normalization(x, mean, variance, beta, gamma, epsilon)

    return activation(norma)
