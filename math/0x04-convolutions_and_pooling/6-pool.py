#!/usr/bin/env python3
""" 0x04. Convolutions and Pooling """
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs pooling on images
    images is a numpy.ndarray with shape (m, h, w, c) containing multiple images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
        c is the number of channels in the image
    kernel_shape is a tuple of (kh, kw) containing the kernel shape for the pooling
        kh is the height of the kernel
        kw is the width of the kernel
    stride is a tuple of (sh, sw)
        sh is the stride for the height of the image
        sw is the stride for the width of the image
    mode indicates the type of pooling
        max indicates max pooling
        avg indicates average pooling
    Returns: a numpy.ndarray containing the pooled images
    """

    x_w, x_h, m = images.shape[2], images.shape[1], images.shape[0]
    c = images.shape[3]
    kernel_w, kernel_h = kernel_shape[1], kernel_shape[0]
    stride_w, stride_h = stride[1], stride[0]

    y_w = int((x_w - kernel_w) / stride_w + 1)
    y_h = int((x_h - kernel_h) / stride_h + 1)

    y = np.zeros((m, y_h, y_w, c))

    if mode == 'avg':
        pooling = np.average
    else:
        pooling = np.max

    for j in range(y_w):
        for i in range(y_h):
            y[:, i, j, :] =\
                pooling(images[:,
                               i * stride_h: i * stride_h + kernel_h,
                               j * stride_w: j * stride_w + kernel_w,
                               :], axis=(1, 2))

    return y
