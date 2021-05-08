#!/usr/bin/env python3

import numpy as np

Deep = __import__('16-deep_neural_network').DeepNeuralNetwork

lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
deep = Deep(X.shape[0], [5, 3, 1])
print(deep.cache)
print(deep.weights)
print(deep.L)
deep.L = 10
print(deep.L)

"""
np.random.seed(0)
deep = Deep(X.shape[0], [])
print(deep.cache)
print(deep.weights)
print(deep.L)
deep.L = 10
print("2", deep.L)

np.random.seed(0)
deep = Deep(X.shape[0], [0])
print(deep.cache)
print(deep.weights)
print(deep.L)
deep.L = 10
print("3", deep.L)

np.random.seed(0)
deep = Deep(X.shape[0], [-1])
print(deep.cache)
print(deep.weights)
print(deep.L)
deep.L = 10
print("4", deep.L)
"""
np.random.seed(0)
deep = Deep(X.shape[0], [1])
print(deep.cache)
print(deep.weights)
print(deep.L)
deep.L = 10
print("5", deep.L)

np.random.seed(0)
deep = Deep(X.shape[0], [5])
print(deep.cache)
print(deep.weights)
print(deep.L)
deep.L = 10
print("6", deep.L)

np.random.seed(0)
deep = Deep(X.shape[0], [1, 3])
print(deep.cache)
print(deep.weights)
print(deep.L)
deep.L = 10
print("7", deep.L)

np.random.seed(0)
deep = Deep(X.shape[0], [1, 3, 1])
print(deep.cache)
print(deep.weights)
print(deep.L)
deep.L = 10
print("8", deep.L)
"""
np.random.seed(0)
deep = Deep(X.shape[0], [5, 0, 1])
print(deep.cache)
print(deep.weights)
print(deep.L)
deep.L = 10
print("9", deep.L)

np.random.seed(0)
deep = Deep(X.shape[0], [5, -1, 1])
print(deep.cache)
print(deep.weights)
print(deep.L)
deep.L = 10
print("10", deep.L)
"""
np.random.seed(0)
deep = Deep(X.shape[0], [5, 3, 1, 5, 3, 1, 5, 3, 1, 5, 3, 1])
print(deep.cache)
print(deep.weights)
print(deep.L)
deep.L = 10
print("11", deep.L)

np.random.seed(0)
deep = Deep(X.shape[0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
print(deep.cache)
print(deep.weights)
print(deep.L)
deep.L = 10
print("12", deep.L)
