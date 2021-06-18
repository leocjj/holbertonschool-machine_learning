#!/usr/bin/env python3
""" 0x08. Deep CNNs """
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
    Builds an inception block as described in Going Deeper with Convolutions
    (2014):

    A_prev is the output from the previous layer
    filters is a tuple or list containing F1, F3R, F3,F5R, F5, FPP,
    respectively:
        F1 is the number of filters in the 1x1 convolution
        F3R is the number of filters in the 1x1 convolution before the 3x3 conv
        F3 is the number of filters in the 3x3 convolution
        F5R is the number of filters in the 1x1 convolution before the 5x5 conv
        F5 is the number of filters in the 5x5 convolution
        FPP is the number of filters in the 1x1 convolution after the max pooli
    All convolutions inside the inception block use a rectified linear
    activation (ReLU)
    Returns: the concatenated output of the inception block
    """

    F1, F3R, F3, F5R, F5, FPP = filters

    c_1x1 = K.layers.Conv2D(F1, 1, activation='relu')(A_prev)
    c_3x3 = K.layers.Conv2D(F3R, 1, activation='relu')(A_prev)
    c_3x3 = K.layers.Conv2D(F3, 3, padding='same', activation='relu')(c_3x3)
    c_5x5 = K.layers.Conv2D(F5R, 1, activation='relu')(A_prev)
    c_5x5 = K.layers.Conv2D(F5, 5, padding='same', activation='relu')(c_5x5)
    pooling = K.layers.MaxPool2D(3, 1, padding='same')(A_prev)
    pooling = K.layers.Conv2D(FPP, 1, activation='relu')(pooling)

    return K.layers.concatenate([c_1x1, c_3x3, c_5x5, pooling])
