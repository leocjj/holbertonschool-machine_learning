#!/usr/bin/env python3
""" 0x04. Convolutions and Pooling """
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images
    images is a numpy.ndarray with shape (m, h, w) containing multiple
        grayscale images:
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
    kernel is a numpy.ndarray with shape (kh, kw) containing the kernel for the
        convolution
        kh is the height of the kernel
        kw is the width of the kernel
    Returns: a numpy.ndarray containing the convolved images
    """

    x_w, x_h, m = images.shape[2], images.shape[1], images.shape[0]
    kernel_w, kernel_h = kernel.shape[1], kernel.shape[0]

    if kernel_h % 2 == 0:
        pad_h = int(kernel_h / 2)
    else:
        pad_h = int((kernel_h - 1) / 2)
    if kernel_w % 2 == 0:
        pad_w = int(kernel_w / 2)
    else:
        pad_w = int((kernel_w - 1) / 2)

    image = np.pad(images, pad_width=((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                   mode='constant')

    y = np.zeros((m, x_h, x_w))

    for j in range(x_w):
        for i in range(x_h):
            y[:, i, j] = (kernel *
                          image[:, i: i + kernel_h,
                                j: j + kernel_w]).sum(axis=(1, 2))

    return y
