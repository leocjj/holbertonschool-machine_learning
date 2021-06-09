#!/usr/bin/env python3
""" 0x07. Convolutional Neural Networks """
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a pooling layer of a neural network:
    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing
        the output of the previous layer
        m is the number of examples
        h_prev is the height of the previous layer
        w_prev is the width of the previous layer
        c_prev is the number of channels in the previous layer
    kernel_shape is a tuple of (kh, kw) containing the size of the kernel for
        the pooling
        kh is the kernel height
        kw is the kernel width
    stride is a tuple of (sh, sw) containing the strides for the pooling
        sh is the stride for the height
        sw is the stride for the width
    mode is a string containing either max or avg, indicating whether to
        perform maximum or average pooling, respectively
    Returns: the output of the pooling layer
    """

    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    ph = int(((h_prev - kh) / sh) + 1)
    pw = int(((w_prev - kw) / sw) + 1)

    conv = np.zeros((m, ph, pw, c_prev))

    if mode == 'max':
        pooling = np.max
    else:
        pooling = np.average

    for i in range(ph):
        for j in range(pw):
            conv[:, i, j, :] = pooling(
                np.reshape(
                    A_prev[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :],
                    (m, kh * kw, c_prev)
                ), axis=1
            )

    return conv
