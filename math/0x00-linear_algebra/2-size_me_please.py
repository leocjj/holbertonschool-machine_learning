#!/usr/bin/env python3
""" calculates the shape of a matrix """


def matrix_shape(matrix):
    """ calculates the shape of a matrix """
    shape = []

    while type(matrix) == list:
        shape.append(len(matrix))
        if type(matrix[0]) == list:
            matrix = matrix[0]
        else:
            break
    return shape
