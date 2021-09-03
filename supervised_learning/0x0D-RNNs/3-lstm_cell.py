#!/usr/bin/env python3
"""
Represents an LSTM unit
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


class LSTMCell:
    """
    represents an LSTM unit
    """
    def __init__(self, i, h, o):
        """
        Class constructor
        """
        self.Wf = np.random.normal(size=(i + h, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))

        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        performs forward propagation for one time step
        Returns: h_next, c_next, y
        """
        concat = np.concatenate((h_prev, x_t), axis=1)

        f_t = sigmoid(concat @ self.Wf + self.bf)
        u_t = sigmoid(concat @ self.Wu + self.bu)
        o_t = sigmoid(concat @ self.Wo + self.bo)
        c_t = np.tanh(concat @ self.Wc + self.bc)
        c_next = f_t * c_prev + u_t * c_t
        h_next = o_t * np.tanh(c_next)
        y = softmax(h_next @ self.Wy + self.by)

        return h_next, c_next, y