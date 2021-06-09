#!/usr/bin/env python3
""" 0x07. Convolutional Neural Networks """
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a convolutional layer of a neural network
    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing
        the output of the previous layer:
        m is the number of examples
        h_prev is the height of the previous layer
        w_prev is the width of the previous layer
        c_prev is the number of channels in the previous layer
    W is a numpy.ndarray of shape (kh, kw, c_prev, c_new) containing the
        kernels for the convolution
        kh is the filter height
        kw is the filter width
        c_prev is the number of channels in the previous layer
        c_new is the number of channels in the output
    b is a numpy.ndarray of shape (1, 1, 1, c_new) containing the biases
        applied to the convolution
    activation is an activation function applied to the convolution
    padding is a string that is either same or valid, indicating the type of
        padding used
    stride is a tuple of (sh, sw) containing the strides for the convolution
        sh is the stride for the height
        sw is the stride for the width
    Returns: the output of the convolutional layer
    """

    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if isinstance(padding, tuple):
        pad_h, pad_w = padding
    elif padding == 'same':
        pad_h = int(((h_prev - 1) * sh + kh - h_prev) / 2)
        pad_w = int(((w_prev - 1) * sw + kw - w_prev) / 2)
    else:
        pad_h, pad_w = 0, 0

    height_conv = int(((h_prev - kh + (2 * pad_h)) / sh) + 1)
    width_conv = int(((w_prev - kw + (2 * pad_w)) / sw) + 1)

    images = np.pad(A_prev, pad_width=((0, 0), (pad_h, pad_h), (pad_w, pad_w),
                                       (0, 0)),
                    mode='constant', constant_values=0)

    conv = np.zeros((m, height_conv, width_conv, c_new))

    for i in range(c_new):
        for j in range(height_conv):
            for k in range(width_conv):
                data = images[:, j * sh:j * sh + kh, k * sw:k * sw + kw] \
                    * W[:, :, :, i]
                conv[:, j, k, i] = np.sum(data, axis=(1, 2, 3)) + b[:, :, :, i]

    return activation(conv)
