#!/usr/bin/env python3
""" 0x07. Convolutional Neural Networks """
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs back propagation over a convolutional layer of a neural network:
    dZ is a numpy.ndarray of shape (m, h_new, w_new, c_new) containing the
        partial derivatives with respect to the unactivated output of the
        convolutional layer:
        m is the number of examples
        h_new is the height of the output
        w_new is the width of the output
        c_new is the number of channels in the output
    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing
        the output of the previous layer
        h_prev is the height of the previous layer
        w_prev is the width of the previous layer
        c_prev is the number of channels in the previous layer
    W is a numpy.ndarray of shape (kh, kw, c_prev, c_new) containing the
        kernels for the convolution
        kh is the filter height
        kw is the filter width
    b is a numpy.ndarray of shape (1, 1, 1, c_new) containing the biases
        applied to the convolution
    padding is a string that is either same or valid, indicating the type of
        padding used
    stride is a tuple of (sh, sw) containing the strides for the convolution
        sh is the stride for the height
        sw is the stride for the width
    Returns: the partial derivatives with respect to the previous layer
        (dA_prev), the kernels (dW), and the biases (db), respectively
    """
    _, h_dp, w_dp, _ = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == 'valid':
        ph, pw = 0, 0

    if padding == 'same':
        ph = int(np.ceil(max((h_prev - 1) * sh + kh - h_prev, 0) / 2))
        pw = int(np.ceil(max((w_prev - 1) * sw + kw - w_prev, 0) / 2))
        A_prev = np.pad(A_prev, pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
                        mode='constant', constant_values=0)

    dW = np.zeros(W.shape)
    dA_prev = np.zeros(A_prev.shape)
    db = np.zeros(b.shape)
    db[:, :, 0, :] = np.sum(dZ, axis=(0, 1, 2))

    for i in range(m):
        for j in range(h_dp):
            for k in range(w_dp):
                for cn in range(c_new):
                    tmp_W = W[:, :, :, cn]
                    tmp_dZ = dZ[i, j, k, cn]
                    dA_prev[i, j * sh:j * sh + kh, k * sw:k * sw + kw] += \
                        (tmp_W * tmp_dZ)
                    dW[:, :, :, cn] += (A_prev[i, j * sh:j * sh + kh,
                                               k * sw:k * sw + kw] * tmp_dZ)

    _, h_dA, w_dA, _ = dA_prev.shape
    dA_prev = dA_prev[:, ph:h_dA-ph, pw:w_dA-pw, :]
    return dA_prev, dW, db
