#!/usr/bin/env python3
""" 0x03. Optimization """
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    updates a variable in place using the Adam optimization algorithm:
    :param alpha: learning rate
    :param beta1: weight used for the first moment
    :param beta2: weight used for the second moment
    :param epsilon: a small number to avoid division by zero
    :param var: ndarray containing the variable to be updated
    :param grad: ndarray containing the gradient of var
    :param v: the previous first moment of var
    :param s: the previous second moment of var
    :param t: the time step used for bias correction
    :return: updated variable, the new first moment, and the new second moment
    """

    v1 = beta1 * v + (1 - beta1) * grad
    s1 = beta2 * s + (1 - beta2) * (grad ** 2)
    v2 = v1 / (1 - (beta1 ** t))
    s2 = s1 / (1 - (beta2 ** t))
    var -= alpha * v2 / (np.sqrt(s2) + epsilon)

    return var, v1, s1
