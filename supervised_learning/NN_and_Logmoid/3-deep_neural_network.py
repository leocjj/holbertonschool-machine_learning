#!/usr/bin/env python3
""" 0x01. Classification """
import numpy as np
import matplotlib.pyplot as plt
from math import exp, log
from pickle import dump, load
from os.path import isfile
#from numba import njit
import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
SIGMOID = 'sigmoid'
TANH = 'tanh'
LOGMOID = 'logmoid'
ACTIVATION_FUNCTIONS = (SIGMOID, TANH, LOGMOID)

#@njit(fastmath=True, parallel=True, nogil=True)
#@njit(fastmath=True)
def sigmoid(x):
    """
    https://stackoverflow.com/questions/3985619/
    how-to-calculate-a-logistic-sigmoid-function-in-python
    Y = 1. / (1. + np.exp(-x))
    :param x: int or array. Use math.ext instead of np.exp for integers.
    :return: sigmoid function of x
    """
    if isinstance(x, int):
        return exp(-np.logaddexp(0., -x))
    else:
        return np.exp(-np.logaddexp(0., -1 * x))

#@njit(fastmath=True, parallel=True, nogil=True)
#@njit(fastmath=True)
def tanh(x):
    """
    :param x: int or array. Use math.ext instead of np.exp for integers.
    :return: tanh function of x
    """
    if isinstance(x, int):
        pos = exp(x)
        neg = exp(-x)
        return (pos - neg) / (pos + neg)
    else:
        return np.tanh(x)

#@njit(fastmath=True, parallel=True, nogil=True)
#@njit(fastmath=True)
def logmoid(z):
    """
    Piecewise-defined function using ln(x+1) for x equals or greater than zero
    and -ln(-x+1) for x less than zero.
    :param z: int or array. Use math.log instead of np.log for integers.
    :return: logmoid function of x
    """
    return np.where(z < 0, -np.log(1 - z), np.log(1 + z))

#@njit(fastmath=True, parallel=True, nogil=True)
#@njit(fastmath=True)
def logmoid_inv(A):
    """
    Piecewise-defined function to calculate the inverse of logmoid.
    i.e.: the output of the neuron before the activation function.
    :param A: int or array. Use math.log instead of np.log for integers.
    :return: inverse logmoid function of A
    """
    return np.where(A < 0, 1 - np.exp(-A), -1 + np.exp(A))


class DeepNeuralNetwork:
    """ Class that defines a deep neural network performing binary
        classification: """

    def __init__(self, nx, layers, activation=SIGMOID):
        """
        Defines a deep neural network performing binary classification.
        :param nx:  is the number of input features.
        :param layers: list with the number of nodes for each layer.
        :param activation: activation function for hidden layers.
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
        if activation not in ACTIVATION_FUNCTIONS:
            raise ValueError("Activation function not implemented.\n" +
                             "Use one of these: {}".format(*ACTIVATION_FUNCTIONS))

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {'W1':
                          np.random.randn(layers[0], nx) * np.sqrt(2 / nx),
                          'b1': np.zeros((layers[0], 1))
                          }
        self.__activation = activation

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

    @property
    def activation(self):
        """  getter method """
        return self.__activation

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network.
        :param X: is a numpy.ndarray with shape (nx, m) that contains the input
            nx is the number of input features to the neuron,
            m is the number of examples.
        :return: the output of the neural network and the cache, respectively.
        """

        self.cache["A0"] = X

        for i in range(1, self.L + 1):
            z = np.matmul(
                          self.weights["W" + str(i)],
                          self.cache["A" + str(i - 1)])\
                + self.weights["b" + str(i)]

            # TODO: select different activation function for output layer.
            # All layers, select activation function:
            if self.__activation == SIGMOID:
                self.cache["A" + str(i)] = sigmoid(z)
            elif self.__activation == TANH:
                self.cache["A" + str(i)] = tanh(z)
            elif self.__activation == LOGMOID:
                self.cache["A" + str(i)] = logmoid(z)

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

        # TODO Evaluate others cost functions.
        if self.__activation == SIGMOID:
            return (-1 / Y.shape[1]) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        elif self.__activation in (TANH, LOGMOID):
            return -1 * np.sum(np.where(Y == 1, np.exp(-A), np.exp(A - 1))) / Y.shape[1]
            # return (-1 / Y.shape[1]) * np.sum(Y * np.exp(-A) + (1 - Y) * np.exp(A - 1))

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
        # TODO: modify - 0.5 dinamically with mean(training_data)
        # TODO: output as a probability (float).
        return np.where(self.cache["A" + str(self.L)] >= 0.5, 1, 0),\
            self.cost(Y, self.cache["A" + str(self.L)])

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron.
        Updates the private attributes __W1, __b1, __W2, and __b2
        :param Y: is a numpy.ndarray with shape (1, m) that contains the
            correct labels for the input data.
        :param cache: is a dictionary containing all the intermediary values
            of the network.
        :param alpha: is the learning rate
        :return: Nothing.
        """

        dZ = cache['A' + str(self.L)] - Y           # ERROR
        m1 = (1 / Y.shape[1])                       # FACTOR TO DIVIDE BY NUMBER OF INPUT DATA

        for i in range(self.L, 0, -1):
            ''' BACKPROPAGATION USING GRADIENT DESCENT'''
            dW = m1 * np.matmul(cache['A' + str(i - 1)], dZ.T)
            db = m1 * np.sum(dZ, axis=1, keepdims=True)
            self.weights['W' + str(i)] -= (alpha * dW).T
            self.weights['b' + str(i)] -= (alpha * db)

            ''' dZ = (Wi * dZ).dA                   ,dA is the activation function derivative.'''
            dZ = np.matmul(self.weights['W' + str(i)].T, dZ)

            # TODO: research second order derivative (Newton’s Method, Hessian Matrix   ).
            # Apply activation function derivative (gradient).
            A = cache['A' + str(i - 1)]             # output of the activation function.
            if self.__activation == SIGMOID:
                dZ *= (A * (1 - A))                 # f'(x) = f(x)(1 - f(x))
            elif self.__activation == TANH:
                dZ *= (1 - np.power(A, 2))          # f'(x) = 1 - f(x)^2
            elif self.__activation == LOGMOID:
                ''' DERIVATIVE OF LOGMOID FUNCTION
                z = np.matmul(self.weights["W" + str(i - 1)], self.cache["A" + str(i - 2)])\
                    + self.weights["b" + str(i - 1)]
                dZ *= np.where(z < 0, np.power(1 - z, -1),  np.power(1 + z, -1))
                z can't be calculated this way for i = 1, because W0 and A-1 don't exist.
                So logmoid_inv is used instead to calculate previous z, i.e. the
                value before the activation function is the inverse of the output
                of the activation function.'''
                z = logmoid_inv(A)                  # f'(x) = {1/(1-x), }
                dZ *= np.where(z < 0, np.power(1 - z, -1), np.power(1 + z, -1))

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """
        Trains the neuron.
        Updates the private attributes __weights and __cache
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
        :param step: verbose and graph will be printed every step iterations.
        :return: the evaluation of the training data after iterations of
            training have occurred
        """

        logging.info('Starting training.')
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
            if not (0 < step <= iterations):
                raise ValueError('step must be positive and <= iterations')

        costs = []
        for i in range(iterations + 1):
            self.forward_prop(X)
            self.gradient_descent(Y, self.cache, alpha)
            if verbose and (i % step == 0 or i == iterations):
                costs.append(self.cost(Y, self.cache["A" + str(self.L)]))
                logging.info("Cost after {} iterations: {}".format(i, costs[-1]))

        if graph:
            plt.plot(np.arange(len(costs)), costs)
            plt.title("Training Cost")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """
        Saves the instance object to a file in pickle format
        :param filename: is the file to which the object should be saved
        :return: None
        """

        if filename == '' or not filename:
            return None
        if not filename.endswith('.pkl'):
            filename += '.pkl'

        with open(filename, 'wb') as f:
            dump(self, f, protocol=3)

    @staticmethod
    def load(filename):
        """
        Loads a pickled DeepNeuralNetwork object.
        :param filename: is the file from which the object should be loaded
        :return: the loaded object, or None if filename doesn’t exist
        """

        if filename == '' or not filename:
            return None
        if not filename.endswith('.pkl'):
            return None
        if not isfile(filename):
            return None

        try:
            f = open(filename, 'rb')
        except IOError:
            return None
        else:
            with f:
                return load(f)
