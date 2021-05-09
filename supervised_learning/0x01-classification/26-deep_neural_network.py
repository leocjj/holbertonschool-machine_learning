#!/usr/bin/env python3
""" 0x01. Classification """
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os.path


def sigmoid(x):
    """
    https://stackoverflow.com/questions/3985619/
    how-to-calculate-a-logistic-sigmoid-function-in-python
    Y = 1. / (1. + np.exp(-x))
    :param x: int or array. Use math.ext instead of np.exp for integers.
    :return: sigmoid function of x
    """
    return np.exp(-np.logaddexp(0., -x))


class DeepNeuralNetwork:
    """ Class that defines a deep neural network performing binary
        classification: """

    def __init__(self, nx, layers):
        """
        Defines a deep neural network performing binary classification.
        :param nx:  is the number of input features.
        :param layers: list with the number of nodes for each layer.
        """
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if not isinstance(layers, list) or not list:
            raise TypeError('layers must be a list of positive integers')
        if not np.issubdtype(np.array(layers).dtype, np.integer) or\
                not all(np.array(layers) >= 1):
            raise TypeError('layers must be a list of positive integers')

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {'W1':
                          np.random.randn(layers[0], nx) * np.sqrt(2 / nx),
                          'b1': np.zeros((layers[0], 1))
                          }

        for i in range(1, self.__L):
            self.__weights["W" + str(i + 1)] =\
                np.random.randn(layers[i], layers[i - 1]) *\
                np.sqrt(2 / layers[i - 1])
            self.__weights["b" + str(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """  getter method """
        return self.__L

    @property
    def cache(self):
        """  getter method """
        return self.__cache

    @property
    def weights(self):
        """  getter method """
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron,
            using a sigmoid activation function.
        :param X: is a numpy.ndarray with shape (nx, m) that contains the input
            nx is the number of input features to the neuron,
            m is the number of examples.
        :return: the output of the neural network and the cache, respectively.
        """

        self.__cache["A0"] = X
        for i in range(1, self.L + 1):
            self.__cache["A" + str(i)] = sigmoid(
                np.matmul(self.weights["W" + str(i)],
                          self.cache["A" + str(i - 1)])
                + self.weights["b" + str(i)]
            )
        return self.cache["A" + str(self.L)], self.cache

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        https://datascience.stackexchange.com/questions/22470/
        python-implementation-of-cost-function-in-logistic-regression-why-dot-multiplic
        :param Y: is a numpy.ndarray with shape (1, m) that contains the
            correct labels for the input data
        :param A: is a numpy.ndarray with shape (1, m) containing the activated
            output of the neuron for each example
        :return: return average of the loss (error) function.
            loss function increase in the opposite sign the output is going.
        """
        return (-1 / Y.shape[1]) *\
            np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))

    def evaluate(self, X, Y):
        """
        Evaluates the neuron’s predictions
        :param X: is a numpy.ndarray with shape (nx, m) that contains the input
            nx is the number of input features to the neuron,
            m is the number of examples.
        :param Y: is a numpy.ndarray with shape (1, m) that contains the
            correct labels for the input data
        :return: tuple with the neuron’s prediction and the cost of the network
            Prediction is numpy.ndarray with shape (1, m) containing the
            predicted labels for each example and the label values should be 1
            if the output of the network is >= 0.5 and 0 otherwise
        """
        self.forward_prop(X)
        return np.heaviside(
            self.cache["A" + str(self.L)] - 0.5, 1
        ).astype(int), self.cost(Y, self.cache["A" + str(self.L)])

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network
        - cache is a dictionary containing all the intermediary values
          of the network
        - alpha is the learning rate
        - Updates the private attribute __weights
        - You are allowed to use one loop
        """
        m = Y.shape[1]
        dZ = self.__cache["A{}".format(self.__L)] - Y
        for i in range(self.__L, 0, -1):
            A = "A{}".format(i-1)
            W = "W{}".format(i)
            b = "b{}".format(i)
            dW = (1/m)*np.matmul(dZ, self.__cache[A].T)
            db = (1/m)*np.sum(dZ, axis=1, keepdims=True)
            dZ = np.matmul(self.__weights[W].T, dZ) * (self.__cache[A] *
                                                       (1 - self.__cache[A]))
            # update of __weights
            self.__weights[W] = self.__weights[W] - alpha*dW
            self.__weights[b] = self.__weights[b] - alpha*db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Trains the deep neural network by updating __weights and __cache
        - You are allowed to use one loop
        - Returns the evaluation of the training data after
          iterations of training have occurred
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose is True or graph is True:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        cost_list = []
        steps_list = []
        for i in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(Y, self.cache, alpha)
            if i % step == 0 or i == iterations:
                cost = self.cost(Y, self.__cache['A{}'.format(self.L)])
                cost_list.append(cost)
                steps_list.append(i)
                if verbose is True:
                    print("Cost after {} iterations: {}".format(i, cost))
        if graph is True:
            plt.plot(steps_list, cost_list, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()
        return self.evaluate(X, Y)

    def save(self, filename):
        """
        Saves the instance object to a file in pickle format
        filename is the file to which the object should be saved
        If filename does not have the extension .pkl, add it
        """
        if filename[-4:] != ".pkl":
            filename = filename + ".pkl"
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        Loads a pickled DeepNeuralNetwork object
        filename is the file from which the object should be loaded
        Returns: the loaded object, or None if filename doesn’t exist
        """
        try:
            # read in binary
            with open(filename, 'rb') as f:
                obj = pickle.load(f)
            return obj
        except FileNotFoundError:
            return None
