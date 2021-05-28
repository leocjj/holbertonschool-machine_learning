#!/usr/bin/env python3
""" 0x05. Regularization """


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Determines if you should stop gradient descent early. Early stopping should
     occur when the validation cost of the network has not decreased relative
     to the optimal validation cost by more than the threshold over a specific
     patience count.
    :param cost: current validation cost of the neural network.
    :param opt_cost: the lowest recorded validation cost of the neural network.
    :param threshold: the threshold used for early stopping.
    :param patience: the patience count used for early stopping
    :param count: the count of how long the threshold has not been met.
    :return: tuple with boolean of whether the network should be stopped early,
     followed by the updated count.
    """

    if cost < opt_cost - threshold:
        return False, 0
    else:
        count += 1
    if count >= patience:
        return True, count
    else:
        return False, count
