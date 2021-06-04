#!/usr/bin/env python3
""" 0x04. Convolutions and Pooling """
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a convolution on grayscale images with custom padding
    images is a numpy.ndarray with shape (m, h, w) containing multiple
        grayscale images:
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
    kernel is a numpy.ndarray with shape (kh, kw) containing the kernel for the
        convolution
        kh is the height of the kernel
        kw is the width of the kernel
    padding is a tuple of (ph, pw)
        ph is the padding for the height of the image
        pw is the padding for the width of the image
        the image should be padded with 0’s
    Returns: a numpy.ndarray containing the convolved images
    """

    x_w, x_h, m = images.shape[2], images.shape[1], images.shape[0]
    kernel_w, kernel_h = kernel.shape[1], kernel.shape[0]
    pad_w, pad_h = padding[1], padding[0]
    y_w = x_w + 2 * pad_w - kernel_w + 1
    y_h = x_h + 2 * pad_h - kernel_h + 1

    image = np.pad(images, pad_width=((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                   mode='constant')

    y = np.zeros((m, y_h, y_w))

    for j in range(y_w):
        for i in range(y_h):
            y[:, i, j] = (kernel *
                          image[:, i: i + kernel_h,
                                j: j + kernel_w]).sum(axis=(1, 2))

    return y
