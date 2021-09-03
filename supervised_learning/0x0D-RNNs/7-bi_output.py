#!/usr/bin/env python3
"""
Represents a bidirectional cell of an RNN
"""
import numpy as np


def softmax(x):
    """
    Softmax function
    """
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


class BidirectionalCell:
    """
    Represents a bidirectional cell of an RNN
    """
    def __init__(self, i, h, o):
        """
        Class constructor
        """
        self.Whf = np.random.normal(size=(i + h, h))
        self.Whb = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h + h, o))
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Calculates the hidden state in the forward direction for one time step.
        Returns: h_next
        """
        concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(concat @ self.Whf + self.bhf)

        return h_next

    def backward(self, h_next, x_t):
        """
        Calculates the hidden state in the backward direction
        for one time step.
        Returns: h_prev
        """
        concat = np.concatenate((h_next, x_t), axis=1)
        h_prev = np.tanh(concat @ self.Whb + self.bhb)

        return h_prev

    def output(self, H):
        """
        Calculates all outputs for the RNN.
        Returns: Y
        """
        t, _, _ = H.shape
        Y = []

        for step in range(t):
            y = softmax(H[step] @ self.Wy + self.by)
            Y.append(y)
        Y = np.array(Y)

        return Y
