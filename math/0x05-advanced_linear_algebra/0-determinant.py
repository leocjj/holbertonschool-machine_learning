#!/usr/bin/env python3
""" 0x05-advanced_linear_algebra """


def determinant(matrix):
    """
    calculates the determinant of a matrix:
    matrix is a list of lists whose determinant should be calculated
    If matrix is not a list of lists, raise a TypeError with the message
        matrix must be a list of lists
    If matrix is not square, raise a ValueError with the message matrix must
        be a square matrix
    The list [[]] represents a 0x0 matrix
    Returns: the determinant of matrix
    """

    if not matrix or not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    for element in matrix:
        if not isinstance(element, list):
            raise TypeError("matrix must be a list of lists")
    if len(matrix) == 1:
        if len(matrix[0]) == 1:
            return matrix[0][0]
        elif matrix == [[]]:
            return 1
    if not matrix[0] or len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")

    t = 0
    for index, _ in enumerate(matrix):
        minor = [[vector for row, vector in enumerate(line) if row != index]
                 for line in matrix[1:]]
        t += (-1) ** index * matrix[0][index] * determinant(minor)
    return t
