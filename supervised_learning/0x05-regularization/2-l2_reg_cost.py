#!/usr/bin/env python3
""" 0x05. Regularization """

import tensorflow as tf


def l2_reg_cost(cost):
    """
    Calculates the cost of a neural network with L2 regularization
    :param cost: tensor containing cost of network without L2 regularization.
    :return: tensor containing cost of network accounting for L2 regularization
    """

    return cost + tf.losses.get_regularization_losses()