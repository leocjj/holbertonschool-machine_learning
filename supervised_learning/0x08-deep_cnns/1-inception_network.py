#!/usr/bin/env python3
""" 0x08. Deep CNNs """
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    Builds inception network as described: Going Deeper with Convolutions 2014
    Input data will have shape (224, 224, 3)
    All convolutions inside and outside the inception block use a rectified
        linear activation (ReLU)
    Use inception_block = __import__('0-inception_block').inception_block
    Returns: the keras model
    """
    
    inputs = K.Input(shape=(224, 224, 3))

    out = K.layers.Conv2D(64, 7, 2, activation='relu', padding='same',
                          input_shape=inputs.shape)(inputs)
    out = K.layers.MaxPool2D(3, 2, padding='same')(out)

    out = K.layers.Conv2D(64, 1, 1, activation='relu', padding='same')(out)
    out = K.layers.Conv2D(192, 3, 1, activation='relu', padding='same')(out)
    out = K.layers.MaxPool2D(3, 2, padding='same')(out)

    """ Inception 3a, 3b """
    out = inception_block(out, [64, 96, 128, 16, 32, 32])
    out = inception_block(out, [128, 128, 192, 32, 96, 64])
    out = K.layers.MaxPool2D(3, 2, padding='same')(out)

    """ Inception 4a, 4b, 4c, 4d, 4e """
    out = inception_block(out, [192, 96, 208, 16, 48, 64])
    out = inception_block(out, [160, 112, 224, 24, 64, 64])
    out = inception_block(out, [128, 128, 256, 24, 64, 64])
    out = inception_block(out, [112, 144, 288, 32, 64, 64])
    out = inception_block(out, [256, 160, 320, 32, 128, 128])
    out = K.layers.MaxPool2D(3, 2, padding='same')(out)

    """ Inception 5a, 5b """
    out = inception_block(out, [256, 160, 320, 32, 128, 128])
    out = inception_block(out, [384, 192, 384, 48, 128, 128])
    out = K.layers.AvgPool2D(7, 1)(out)

    out = K.layers.Dropout(.4)(out)
    """ Linear softmax """
    out = K.layers.Dense(1000)(out)

    return K.Model(inputs, out)
