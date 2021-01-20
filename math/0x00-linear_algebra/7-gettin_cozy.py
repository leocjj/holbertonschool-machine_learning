#!/usr/bin/env python3
""" 0x00-linear_algebra Task 7 """


def cat_matrices2D(mat1, mat2, axis=0):
    """ concatenates two matrices along a specific axis. """
    if axis == 0 and len(mat1[0]) == len(mat2[0]):
        # return [*deepcopy(mat1), *deepcopy(mat2)]
        up_mat = [sublist[:] for sublist in mat1]
        down_mat = [sublist[:] for sublist in mat2]
        return [*up_mat, *down_mat]
    elif axis == 1 and len(mat1) == len(mat2):
        return [[*mat1[i], *mat2[i]] for i in range(len(mat1))]
    else:
        return None
