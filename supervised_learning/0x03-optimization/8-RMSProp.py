#!/usr/bin/env python3
""" 0x03. Optimization """
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    Creates the training operation for a neural network in tensorflow using
    the RMSProp optimization algorithm.
    :param loss: loss of the network
    :param alpha: learning rate
    :param beta2: RMSProp weight
    :param epsilon: a small number to avoid division by zero
    :return: RMSProp optimization operation
    """

    rms = tf.train.RMSPropOptimizer(alpha, beta2, epsilon=epsilon)
    compute_gradients = rms.compute_gradients(loss)
    apply_gradients = rms.apply_gradients(compute_gradients)

    return apply_gradients
