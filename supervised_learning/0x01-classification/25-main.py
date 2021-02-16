#!/usr/bin/env python3

import numpy as np

oh_encode = __import__('24-one_hot_encode').one_hot_encode
oh_decode = __import__('25-one_hot_decode').one_hot_decode

#lib = np.load('../data/MNIST.npz')
#Y = lib['Y_train'][:10]

Y= np.array([5, 0, 4, 1, 9, 2, 1, 3, 1, 4])
print(Y)
Y_one_hot = oh_encode(Y, 10)
Y_decoded = oh_decode(Y_one_hot)
print(Y_decoded)

Y_one_hot = np.array([])
Y_decoded = oh_decode(Y_one_hot)
print(Y_decoded)

Y_one_hot = np.array([[]])
Y_decoded = oh_decode(Y_one_hot)
print(Y_decoded)

Y_one_hot = np.array([[-1]])
Y_decoded = oh_decode(Y_one_hot)
print(Y_decoded)

Y_one_hot = {}
Y_decoded = oh_decode(Y_one_hot)
print(Y_decoded)

Y_one_hot = 'a'
Y_decoded = oh_decode(Y_one_hot)
print(Y_decoded)

Y_one_hot = None
Y_decoded = oh_decode(Y_one_hot)
print(Y_decoded)

Y_one_hot = np.array([[1]])
Y_decoded = oh_decode(Y_one_hot)
print(Y_decoded)

Y_one_hot = np.array([[2]])
Y_decoded = oh_decode(Y_one_hot)
print(Y_decoded)

Y_one_hot = np.array([[1]])
Y_decoded = oh_decode(Y_one_hot)
print(Y_decoded)
