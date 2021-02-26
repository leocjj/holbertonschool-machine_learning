#!/usr/bin/env python3
""" 0x03. Optimization """
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Updates a variable using the RMSProp optimization algorithm:
    :param alpha: learning rate
    :param beta2: RMSProp weight
    :param epsilon: a small number to avoid division by zero
    :param var: ndarray containing the variable to be updated
    :param grad: ndarray containing the gradient of var
    :param s: the previous second moment of var
    :return: updated variable and the new moment
    """

    s_next = beta2 * s + (1 - beta2) * (grad ** 2)
    var = var - alpha * (grad / (np.sqrt(s_next) + epsilon))

    return var, s_next
