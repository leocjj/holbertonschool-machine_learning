#!/usr/bin/env python3
""" 0x03. Optimization """
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Creates a learning rate decay operation in tensorflow using inverse time
    decay. The learning rate decay should occur in a stepwise fashion.
    :param alpha: original learning rate
    :param decay_rate: used to determine the rate at which alpha will decay
    :param global_step: number of passes of gradient descent that have elapsed
    :param decay_step: number of passes of gradient descent that should occur
        before alpha is decayed further
    :return: learning rate decay operation
    """

    lrd_op = tf.train.inverse_time_decay(alpha, global_step, decay_step,
                                         decay_rate, staircase=True)

    return lrd_op
