#!/usr/bin/env python3
import numpy as np
import time


Deep = __import__('3-deep_neural_network').DeepNeuralNetwork
ACTIVATION_FUNCTIONS = __import__('3-deep_neural_network').ACTIVATION_FUNCTIONS

# DATA LOAD
lib_train = np.load('Binary_Train.npz')
X_train_3D, Y_train = lib_train['X'], lib_train['Y']
X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T
lib_dev = np.load('Binary_Dev.npz')
X_dev_3D, Y_dev = lib_dev['X'], lib_dev['Y']
X_dev = X_dev_3D.reshape((X_dev_3D.shape[0], -1)).T
np.random.seed(0)
network_layers = [100, 100, 100, 100, 1]  # hidden and output layers.


def train_nn_with_activation_function(activation_function, iterations):
    # NETWORK INSTANCE
    deep = Deep(X_train.shape[0], network_layers, activation=activation_function)

    # NETWORK TRAINING
    time1 = time.time()
    A_train, training_cost = deep.train(X_train, Y_train, alpha=0.05, verbose=False,
                         iterations=iterations, step=1)
    time2 = time.time()

    # TRAINING DATA REPORT
    training_accuracy = np.sum(A_train == Y_train) / Y_train.shape[1] * 100
    print("********** REPORT FOR {} **********".format(activation_function))
    print("Iterations: {}".format(iterations))
    # print("Train cost: {}".format(round(training_cost, 4)))
    # print("Train accuracy: {}%".format(round(training_accuracy, 4)))

    # DEV DATA REPORT
    A_dev, dev_cost = deep.evaluate(X_dev, Y_dev)
    dev_accuracy = np.sum(A_dev == Y_dev) / Y_dev.shape[1] * 100
    print("Dev cost: {}".format(round(dev_cost, 4)))
    print("Dev accuracy: {}%".format(round(dev_accuracy, 4)))
    print("Training time: {} seconds\n".format(round(time2 - time1, 0)))

    return (
        iterations,
        round(dev_cost, 4),
        round(dev_accuracy, 4),
        round(time2 - time1, 4)
    )

if __name__ == "__main__":
    print("Network layout: {}\n".format(network_layers))
    #train_nn_with_activation_function(ACTIVATION_FUNCTIONS[0], 5)
    #train_nn_with_activation_function(ACTIVATION_FUNCTIONS[1], 5)
    #train_nn_with_activation_function(ACTIVATION_FUNCTIONS[2], 5)

    results = []
    for iterations in range(5, 21, 5):
        for function in ACTIVATION_FUNCTIONS:
            results.append(train_nn_with_activation_function(function, iterations))
