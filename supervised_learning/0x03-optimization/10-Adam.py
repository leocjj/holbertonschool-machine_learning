#!/usr/bin/env python3
""" 0x03. Optimization """
import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    creates the training operation for a neural network in tensorflow using
        the Adam optimization algorithm:
    :param loss: loss of the network
    :param alpha: learning rate
    :param beta1: weight used for the first moment
    :param beta2: weight used for the second moment
    :param epsilon: a small number to avoid division by zero
    :return: Adam optimization operation
    """

    optimizer = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon)
    adam = optimizer.minimize(loss)

    return adam
