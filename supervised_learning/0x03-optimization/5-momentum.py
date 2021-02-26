#!/usr/bin/env python3
""" 0x03. Optimization """


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    updates a variable using the gradient descent with momentum optimization:
    :param alpha: the learning rate
    :param beta1: the momentum weight
    :param var: ndarray containing the variable to be updated
    :param grad: ndarray containing the gradient of var
    :param v: the previous first moment of var
    :return: updated variable and the new moment
    """

    v_next = beta1 * v + (1 - beta1) * grad
    var_next = var - (alpha * v_next)
    return var_next, v_next
