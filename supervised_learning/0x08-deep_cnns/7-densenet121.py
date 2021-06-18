#!/usr/bin/env python3
""" 0x08. Deep CNNs """


import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    builds the DenseNet-121 architecture as described in Densely Connected
        Convolutional Networks:
    growth_rate is the growth rate
    compression is the compression factor
    You can assume the input data will have shape (224, 224, 3)
    All convolutions should be preceded by Batch Normalization and a rectified
        linear activation (ReLU), respectively
    All weights use he normal initialization
    Use:
        dense_block = __impor__('5-dense_block').dense_block
        transition_layer = __impor__('6-transition_layer').transition_layer
    Returns: the keras model
    """
    inputs = K.Input(shape=(224, 224, 3))
    out_l = K.layers.BatchNormalization(axis=3)(inputs)
    out_l = K.layers.Activation('relu')(out_l)
    out_l = K.layers.Conv2D(64, kernel_size=(7, 7), padding='same',
                            kernel_initializer='he_normal',
                            strides=(2, 2))(out_l)
    out_l = K.layers.MaxPool2D((3, 3), (2, 2), padding="same")(out_l)
    out_l, filters = dense_block(out_l, 64, growth_rate, 6)
    out_l, filters = transition_layer(out_l, filters, compression)
    out_l, filters = dense_block(out_l, filters, growth_rate, 12)
    out_l, filters = transition_layer(out_l, filters, compression)
    out_l, filters = dense_block(out_l, filters, growth_rate, 24)
    out_l, filters = transition_layer(out_l, filters, compression)
    out_l, filters = dense_block(out_l, filters, growth_rate, 16)
    out_l = K.layers.AvgPool2D((7, 7), padding='same')(out_l)
    out_l = K.layers.Dense(1000, activation='softmax')(out_l)
    model = K.Model(inputs, out_l)
    return model
