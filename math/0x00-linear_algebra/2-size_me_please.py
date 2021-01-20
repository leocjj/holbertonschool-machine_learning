#!/usr/bin/env python3
def matrix_shape(matrix):
    """ Calculates the shape of a matrix
    Args:
        matrix ([int]): contains all elements
    """
    rows = len(matrix)
    items = 0
    columns = 0
    shape = []

    for item in matrix:
        columns = len(item)
        for sub_item in item:
            if isinstance(sub_item, list):
                items = len(sub_item)

    shape.append(rows)
    shape.append(columns)
    if items != 0:
        shape.append(items)
    return shape
