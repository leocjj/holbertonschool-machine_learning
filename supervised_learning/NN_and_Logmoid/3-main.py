#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import time
import logging


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
Deep = __import__('3-deep_neural_network').DeepNeuralNetwork
ACTIVATION_FUNCTIONS = __import__('3-deep_neural_network').ACTIVATION_FUNCTIONS

# DATA LOAD
lib_train = np.load('../data/Binary_Train.npz')
X_train_3D, Y_train = lib_train['X'], lib_train['Y']
X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T
lib_dev = np.load('../data/Binary_Dev.npz')
X_dev_3D, Y_dev = lib_dev['X'], lib_dev['Y']
X_dev = X_dev_3D.reshape((X_dev_3D.shape[0], -1)).T
np.random.seed(0)

# NETWORK INSTANCE
network_layers = [5, 3, 1]           # hidden and output layers.
activation_function = ACTIVATION_FUNCTIONS[2]
deep = Deep(X_train.shape[0], network_layers, activation=activation_function)

# NETWORK TRAINING
print("\n***************************** TRAINING ***************************\n")
time1 = time.time()
A, cost = deep.train(X_train, Y_train, alpha=0.1, iterations=5, step=1)
time2 = time.time()

# TRAINING DATA REPORT
accuracy = np.sum(A == Y_train) / Y_train.shape[1] * 100
print("\n*************************** FINAL REPORT *************************\n")
logging.info("Network layout: {}".format(network_layers))
logging.info("Activation function: {}".format(activation_function))
logging.info("Train cost: {}".format(cost))
logging.info("Train accuracy: {}%\n".format(accuracy))

# TEST DATA REPORT
A, cost = deep.evaluate(X_dev, Y_dev)
accuracy = np.sum(A == Y_dev) / Y_dev.shape[1] * 100
logging.info("Dev cost: {}".format(cost))
logging.info("Dev accuracy: {}%\n".format(accuracy))

logging.info("Training time: {} seconds".format(time2 - time1))

# PLOT WITH IMAGES AND OUTPUT
fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_dev_3D[i])
    plt.title(A[0, i])
    plt.axis('off')
plt.tight_layout()
plt.show()
