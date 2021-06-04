#!/usr/bin/env python3
""" 0x04. Convolutions and Pooling """
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    performs a valid convolution on grayscale images
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
    y_h = x_h - kernel_h + 1
    y_w = x_w - kernel_w + 1

    y = np.zeros((m, y_h, y_w))

    for j in range(y_w):
        for i in range(y_h):
            y[:, i, j] = (kernel *
                          images[:, i: i + kernel_h,
                                 j: j + kernel_w]).sum(axis=(1, 2))

    return y
