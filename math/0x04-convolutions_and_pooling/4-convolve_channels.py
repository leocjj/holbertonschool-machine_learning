#!/usr/bin/env python3
""" 0x04. Convolutions and Pooling """
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    images is a numpy.ndarray with shape (m, h, w, c) containing multiple images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
        c is the number of channels in the image
    kernel is a numpy.ndarray with shape (kh, kw, c) containing the kernel for
        the convolution
        kh is the height of the kernel
        kw is the width of the kernel
    padding is either a tuple of (ph, pw), ‘same’, or ‘valid’
        if ‘same’, performs a same convolution
        if ‘valid’, performs a valid convolution
        if a tuple:
            ph is the padding for the height of the image
            pw is the padding for the width of the image
        the image should be padded with 0’s
    stride is a tuple of (sh, sw)
        sh is the stride for the height of the image
        sw is the stride for the width of the image
    Returns: a numpy.ndarray containing the convolved images
    """

    x_w, x_h, m = images.shape[2], images.shape[1], images.shape[0]
    kernel_w, kernel_h = kernel.shape[1], kernel.shape[0]
    stride_w, stride_h = stride[1], stride[0]

    if isinstance(padding, tuple):
        pad_w, pad_h = padding[1], padding[0]
    elif padding == 'same':
        pad_h = int(((x_h - 1) * stride_h + kernel_h - x_h) / 2) + 1
        pad_w = int(((x_w - 1) * stride_w + kernel_w - x_w) / 2) + 1
    else:
        pad_h = 0
        pad_w = 0

    y_w = int((x_w + 2 * pad_w - kernel_w) / stride_w + 1)
    y_h = int((x_h + 2 * pad_h - kernel_h) / stride_h + 1)

    image = np.pad(images,
                   pad_width=((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
                   mode='constant')

    y = np.zeros((m, y_h, y_w))

    for j in range(y_w):
        for i in range(y_h):
            y[:, i, j] = (kernel *
                          image[:,
                                i * stride_h: i * stride_h + kernel_h,
                                j * stride_w: j * stride_w + kernel_w]
                          ).sum(axis=(1, 2, 3))

    return y
