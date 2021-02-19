#!/usr/bin/env python3
""" 0x02. Tensorflow """


create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Function that creates the forward propagation graph for the neural network.
    :param x: is the placeholder for input data
    :param layer_sizes: is a list containing the number of nodes in each layer.
    :param activations: a list containing activation function for each layer.
    :return: the prediction of the network in tensor form
    """

    layer = create_layer(x, layer_sizes[0], activations[0])
    for i in range(1, len(layer_sizes)):
        layer = create_layer(layer, layer_sizes[i], activations[i])

    return layer
