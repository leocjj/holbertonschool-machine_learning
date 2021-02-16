#!/usr/bin/env python3

import numpy as np

oh_encode = __import__('24-one_hot_encode').one_hot_encode

lib = np.load('../data/MNIST.npz')
Y = lib['Y_train'][:10]

print(Y)
Y_one_hot = oh_encode(Y, 10)
print(Y_one_hot)

Y = np.array([])
print(Y)
Y_one_hot = oh_encode(Y, 10)
print(Y_one_hot)

Y = np.array([10])
print(Y)
Y_one_hot = oh_encode(Y, 0)
print(Y_one_hot)

Y = np.array([])
print(Y)
Y_one_hot = oh_encode(Y, 0)
print(Y_one_hot)

Y = np.array({})
print(Y)
Y_one_hot = oh_encode(Y, 10)
print(Y_one_hot)

Y = 5
print(Y)
Y_one_hot = oh_encode(Y, 10)
print(Y_one_hot)

Y = (1,)
print(Y)
Y_one_hot = oh_encode(Y, 10)
print(Y_one_hot)

Y = 'A'
print(Y)
Y_one_hot = oh_encode(Y, 10)
print(Y_one_hot)

Y = [-1]
print(Y)
Y_one_hot = oh_encode(Y, 1)
print(Y_one_hot)

Y = [1]
print(Y)
Y_one_hot = oh_encode(Y, -1)
print(Y_one_hot)

Y = [-1]
print(Y)
Y_one_hot = oh_encode(Y, -1)
print(Y_one_hot)

Y = [2]
print(Y)
Y_one_hot = oh_encode(Y, 2)
print(Y_one_hot)

Y = np.array([2])
print(Y)
Y_one_hot = oh_encode(Y, 2)
print(Y_one_hot)

Y = np.array([2])
print(Y)
Y_one_hot = oh_encode(Y, 1)
print(Y_one_hot)

Y = np.array([1])
print(Y)
Y_one_hot = oh_encode(Y, 2)
print(Y_one_hot)

Y = np.array([5])
print(Y)
Y_one_hot = oh_encode(Y, 100)
print(Y_one_hot)

Y = np.array([100])
print(Y)
Y_one_hot = oh_encode(Y, 5)
print(Y_one_hot)

Y = np.array(['a'])
print(Y)
Y_one_hot = oh_encode(Y, 10)
print(Y_one_hot)

Y = np.array([[2]])
print(Y)
Y_one_hot = oh_encode(Y, 10)
print(Y_one_hot)

Y = np.array([50])
print(Y)
Y_one_hot = oh_encode(Y, 50)
print(Y_one_hot)

Y = np.array([1, 1, 2, 3, 4, 5])
print(Y)
Y_one_hot = oh_encode(Y, 6)
print(Y_one_hot)

Y = np.array([1, 1, 2, 3, 4, 5])
print(Y)
Y_one_hot = oh_encode(Y, -1)
print(Y_one_hot)

Y = np.array([1, 1, 2, 3, 4, 5])
print(Y)
Y_one_hot = oh_encode(Y, 0)
print(Y_one_hot)

Y = np.array([1, 1, 2, 3, 4, 5])
print(Y)
Y_one_hot = oh_encode(Y, 'A')
print(Y_one_hot)

Y = np.array([1, 1, 2, 3, 4, 5])
print(Y)
Y_one_hot = oh_encode(Y, [])
print(Y_one_hot)

Y = np.array([1, 1, 2, 3, 4, 5])
print(Y)
Y_one_hot = oh_encode(Y, {})
print(Y_one_hot)
