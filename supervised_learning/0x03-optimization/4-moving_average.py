#!/usr/bin/env python3
""" 0x03. Optimization """


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a data set.
    Moving average calculation should use bias correction.
    :param data: list of data to calculate the moving average of
    :param beta: weight used for the moving average
    :return: list containing the moving averages of data
    """

    temp = 0
    average = []
    for i, val_i in enumerate(data):
        temp = (beta * temp) + ((1 - beta) * val_i)
        mv = temp / (1 - beta ** (i + 1))
        average.append(mv)
    return average
