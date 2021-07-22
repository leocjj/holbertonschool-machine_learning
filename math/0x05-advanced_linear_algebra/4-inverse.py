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
        min = [[vector for row, vector in enumerate(line) if row != index]
               for line in matrix[1:]]
        t += (-1) ** index * matrix[0][index] * determinant(min)
    return t


def minor(matrix):
    """
    calculates the minor matrix of a matrix:

    matrix is a list of lists whose minor matrix should be calculated
    If matrix is not a list of lists, raise a TypeError with the message
        matrix must be a list of lists
    If matrix is not square or is empty, raise a ValueError with the message
        matrix must be a non-empty square matrix
    Returns: the minor matrix of matrix
    """

    if not matrix or not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    for element in matrix:
        if not isinstance(element, list):
            raise TypeError("matrix must be a list of lists")
    if matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")
    for row in matrix:
        if len(row) != len(matrix):
            raise ValueError("matrix must be a non-empty square matrix")

    if len(matrix) == 1:
        return [[1]]

    min = []
    for i, _ in enumerate(matrix):
        min.append([])
        for j, _ in enumerate(matrix):
            rows = [matrix[m] for m, _ in enumerate(matrix) if m != i]
            new_m = [[row[n] for n, _ in enumerate(matrix) if n != j]
                     for row in rows]
            my_det = determinant(new_m)
            min[i].append(my_det)

    return min


def cofactor(matrix):
    """
    calculates the cofactor matrix of a matrix:

    matrix is a list of lists whose cofactor matrix should be calculated
    If matrix is not a list of lists, raise a TypeError with the message matrix
        must be a list of lists
    If matrix is not square or is empty, raise a ValueError with the message
        matrix must be a non-empty square matrix
    Returns: the cofactor matrix of matrix
    """

    if not matrix or not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    for element in matrix:
        if not isinstance(element, list):
            raise TypeError("matrix must be a list of lists")
    if matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")
    for row in matrix:
        if len(row) != len(matrix):
            raise ValueError("matrix must be a non-empty square matrix")

    if len(matrix) == 1:
        return [[1]]

    return [[(-1) ** (i + j) * minor(matrix)[i][j]
             for j in range(len(matrix))]
            for i in range(len(matrix))]


def adjugate(matrix):
    """
    calculates the adjugate matrix of a matrix:

    matrix is a list of lists whose adjugate matrix should be calculated
    If matrix is not a list of lists, raise a TypeError with the message matrix
        must be a list of lists
    If matrix is not square or is empty, raise a ValueError with the message
        matrix must be a non-empty square matrix
    Returns: the adjugate matrix of matrix
    """

    if not matrix or not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    for element in matrix:
        if not isinstance(element, list):
            raise TypeError("matrix must be a list of lists")
    if matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")
    for row in matrix:
        if len(row) != len(matrix):
            raise ValueError("matrix must be a non-empty square matrix")

    return [[cofactor(matrix)[j][i]
             for j in range(len(matrix))]
            for i in range(len(matrix))]


def inverse(matrix):
    """
    calculates the inverse of a matrix:

    matrix is a list of lists whose inverse should be calculated
    If matrix is not a list of lists, raise a TypeError with the message matrix
        must be a list of lists
    If matrix is not square or is empty, raise a ValueError with the message
        matrix must be a non-empty square matrix
    Returns: the inverse of matrix, or None if matrix is singular
    """

    if not matrix or not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    for element in matrix:
        if not isinstance(element, list):
            raise TypeError("matrix must be a list of lists")
    if matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")
    for row in matrix:
        if len(row) != len(matrix):
            raise ValueError("matrix must be a non-empty square matrix")

    det = determinant(matrix)
    if det == 0:
        return None

    return [[n / det for n in row] for row in adjugate(matrix)]
