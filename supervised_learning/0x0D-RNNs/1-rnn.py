#!/usr/bin/env python3
"""
Performs forward propagation for a simple RNN
"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Performs forward propagation for a simple RNN
    Returns: H, Y
    """
    t, _, _ = X.shape

    H = []
    Y = []
    H.append(h_0)

    for step in range(t):
        h_next, y = rnn_cell.forward(H[-1], X[step])
        H.append(h_next)
        Y.append(y)

    H = np.array(H)
    Y = np.array(Y)

    return H, Y
