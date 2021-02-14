#!/usr/bin/env python3
""" 0x01. Classification """
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """
    https://stackoverflow.com/questions/3985619/
    how-to-calculate-a-logistic-sigmoid-function-in-python
    Y = 1. / (1. + np.exp(-x))
    :param x: int or array. Use math.ext instead of np.exp for integers.
    :return: sigmoid function of x
    """
    return np.exp(-np.logaddexp(0., -x))


class Neuron:
    """ Class that defines a single neuron performing binary classification """

    def __init__(self, nx):
        """
        Defines a single neuron performing binary classification
        :param nx: number of input features to the neuron
        """
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """  getter method """
        return self.__W

    @property
    def b(self):
        """  getter method """
        return self.__b

    @property
    def A(self):
        """  getter method """
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron,
            using a sigmoid activation function.
        :param X: is a numpy.ndarray with shape (nx, m) that contains the input
            nx is the number of input features to the neuron,
            m is the number of examples.
        :return: the private attribute __A
        """
        self.__A = sigmoid(np.matmul(self.W, X) + self.b)
        return self.A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        https://datascience.stackexchange.com/questions/22470/
        python-implementation-of-cost-function-in-logistic-regression-why-dot-multiplic
        :param Y: is a numpy.ndarray with shape (1, m) that contains the
            correct labels for the input data
        :param A: is a numpy.ndarray with shape (1, m) containing the activated
            output of the neuron for each example
        :return: return -1 / Y.shape[1] * np.sum( np.multiply(np.log(A), Y) +
            np.multiply(np.log(1.0000001 - A), (1.0000001 - Y)))
        """
        return -1 / Y.shape[1] * np.sum(
            np.multiply(np.log(A), Y) +
            np.multiply(np.log(1.0000001 - A), (1 - Y))
        )

    def evaluate(self, X, Y):
        """
        Evaluates the neuron’s predictions
        :param X: is a numpy.ndarray with shape (nx, m) that contains the input
            nx is the number of input features to the neuron,
            m is the number of examples.
        :param Y: is a numpy.ndarray with shape (1, m) that contains the
            correct labels for the input data
        :return: tuple with the neuron’s prediction and the cost of the network
            prediction is numpy.ndarray with shape (1, m) containing the
            predicted labels for each example and the label values should be 1
            if the output of the network is >= 0.5 and 0 otherwise
        """
        self.forward_prop(X)
        return np.heaviside(self.A - 0.5, 1).astype(int), self.cost(Y, self.A)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron.
        Updates the private attributes __W and __b.
        :param X: is a numpy.ndarray with shape (nx, m) that contains the input
            nx is the number of input features to the neuron,
            m is the number of examples.
        :param Y: is a numpy.ndarray with shape (1, m) that contains the
            correct labels for the input data
        :param A: is a numpy.ndarray with shape (1, m) containing the activated
            output of the neuron for each example
        :param alpha: is the learning rate
        :return: Nothing.
        """
        m = Y.shape[1]
        dZ = A - Y
        self.__W -= (alpha * ((1 / m) * np.matmul(X, dZ.T))).T
        self.__b -= (alpha * ((1 / m) * np.sum(dZ)))

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """
        Trains the neuron.
        Updates the private attributes __W, __b, and __A.
        :param X: is a numpy.ndarray with shape (nx, m) that contains the input
            nx is the number of input features to the neuron,
            m is the number of examples.
        :param Y: is a numpy.ndarray with shape (1, m) that contains the
            correct labels for the input data
        :param iterations: is the number of iterations to train over
        :param alpha: is the learning rate
        :param verbose: is a boolean, print information about the training.
        :param graph: is a boolean that defines whether or not to graph
            information about the training once the training has completed.
        :param step:

        :return: the evaluation of the training data after iterations of
            training have occurred
        """
        if not isinstance(iterations, int):
            raise TypeError('iterations must be an integer')
        if iterations < 1:
            raise ValueError('iterations must be a positive integer')
        if not isinstance(alpha, float):
            raise TypeError('alpha must be a float')
        if alpha < 0:
            raise ValueError('alpha must be positive')
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError('step must be an integer')
            if not (1 < iterations <= iterations):
                raise ValueError('step must be positive and <= iterations')

        costs = []
        steps = np.arange(0, iterations + step, step)
        for i in range(iterations + 1):
            self.forward_prop(X)
            if verbose and i % step == 0:
                cost = self.cost(Y, self.__A)
                print("Cost after {} iterations: {}".format(i, cost))
                costs.append(cost)
            self.gradient_descent(X, Y, self.__A, alpha)

        if graph:
            plt.plot(steps, costs)
            plt.title("Training Cost")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.show()
        return self.evaluate(X, Y)
