#!/usr/bin/env python3
""" 0x03. Optimization """
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """
    updates a variable using the gradient descent with momentum optimization:
    :param loss: the loss of the network
    :param alpha: the learning rate
    :param beta1: the momentum weight
    :return: momentum optimization operation
    """

    momentum = tf.train.MomentumOptimizer(alpha, beta1)
    compute_gradients = momentum.compute_gradients(loss)
    apply_gradients = momentum.apply_gradients(compute_gradients)

    return apply_gradients
