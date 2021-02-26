#!/usr/bin/env python3
""" 0x03. Optimization """
import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates the learning rate using inverse time decay in numpy:
    The learning rate decay should occur in a stepwise fashion.
    :param alpha: learning rate
    :param decay_rate: used to determine the rate at which alpha will decay
    :param global_step: number of passes of gradient descent that have elapsed
    :param decay_step: number of passes of gradient descent that should occur
        before alpha is decayed further
    :return: updated value for alpha
    """

    alpha /= 1 + decay_rate * np.floor(global_step / decay_step)
    return alpha
