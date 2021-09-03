#!/usr/bin/env python3
"""
Represents a cell of a simple RNN
"""
import numpy as np


class RNNCell:
    """
    represents a cell of a simple RNN
    """
    def __init__(self, i, h, o):
        """
        Class constructor
        """
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step.
        Returns: h_next, y
        """
        concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(concat @ self.Wh + self.bh)
        soft = h_next @ self.Wy + self.by
        y = np.exp(soft) / np.sum(np.exp(soft), axis=1, keepdims=True)

        return h_next, y
