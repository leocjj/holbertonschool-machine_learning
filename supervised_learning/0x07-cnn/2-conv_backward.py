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

    img_n, _, _, img_c_out = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if isinstance(padding, tuple):
        ph, pw = padding
    elif padding == "same":
        ph = int(((h_prev - 1) * sh + kh - h_prev) / 2)
        pw = int(((w_prev - 1) * sw + kw - w_prev) / 2)
    else:
        ph, pw = 0, 0

    h_prev_out = int((h_prev + 2 * ph - kh) / stride[0]) + 1
    w_prev_out = int((w_prev + 2 * pw - kw) / stride[1]) + 1

    pad = np.pad(A_prev, pad_width=((0, 0), (ph, ph,), (pw, pw), (0, 0)),
                 mode="constant", constant_values=0)
    dW = np.zeros(W.shape)
    dA = np.zeros(pad.shape)

    for i in range(0, img_n):
        for h in range(0, h_prev_out):
            for w in range(0, w_prev_out):
                for c in range(0, img_c_out):
                    dA_slice = dA[i, h * stride[0]: h * stride[0] + kh,
                                  w * stride[1]: w * stride[1] + kw, :]
                    Ap_s = pad[i, h * stride[0]: h * stride[0] + kh,
                               w * stride[1]: w * stride[1] + kw, :]
                    dA_slice += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += Ap_s * dZ[i, h, w, c]
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    return dA, dW, db
