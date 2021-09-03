#!/usr/bin/env python3
"""
Performs forward propagation for a deep RNN
"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Returns: H, Y
    """
    t, m, _ = X.shape
    l, _, h = h_0.shape
    H = np.zeros((t + 1, l, m, h))
    H[0, :, :, :] = h_0
    Y = []

    for step in range(t):
        for layer in range(l):
            if layer == 0:
                h_next, y = rnn_cells[layer].forward(H[step, layer], X[step])
            else:
                h_next, y = rnn_cells[layer].forward(H[step, layer], h_next)
            H[step + 1, layer] = h_next
        Y.append(y)
    Y = np.array(Y)

    return H, Y