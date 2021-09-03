#!/usr/bin/env python3
"""
Represents a gated recurrent unit
"""
import numpy as np


def sigmoid(x):
    """
    Sigmoid function
    """
    return 1 / (1 + np.exp(-x))


def softmax(x):
    """
    Softmax function
    """
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


class GRUCell:
    """
    Represents a gated recurrent unit
    """
    def __init__(self, i, h, o):
        """
        Class constructor
        """
        self.Wz = np.random.normal(size=(i + h, h))
        self.Wr = np.random.normal(size=(i + h, h))
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))

        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step.
        Returns: h_next, y
        """
        concat = np.concatenate((h_prev, x_t), axis=1)
        z_t = sigmoid(concat @ self.Wz + self.bz)
        r_t = sigmoid(concat @ self.Wr + self.br)
        intermediate = np.concatenate((r_t * h_prev, x_t), axis=1)
        h_t = np.tanh(intermediate @ self.Wh + self.bh)
        h_next = (1 - z_t) * h_prev + z_t * h_t
        y = softmax(h_next @ self.Wy + self.by)

        return h_next, y
