#!/usr/bin/env python3
"""
0x04. Autoencoders
"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    creates a convolutional autoencoder:
    input_dims a tuple of integers containing the dimensions of the model input
    filters is a list containing the number of filters for each convolutional
        layer in the encoder, respectively
        the filters should be reversed for the decoder
    latent_dims is a tuple of integers containing the dimensions of the latent
        space representation
    Each convolution in the encoder should use a kernel size of (3, 3) with
        same padding and relu activation, followed by max pooling of size (2,2)
    Each convolution in the decoder, except for the last two, should use a
        filter size of (3, 3) with same padding and relu activation,
        followed by upsampling of size (2, 2)
        Second to last convolution should instead use valid padding
        Last convolution should have the same number of filters as the number
        of channels in input_dims with sigmoid activation and no upsampling.
    Returns: encoder, decoder, auto
        encoder is the encoder model
        decoder is the decoder model
        auto is the full autoencoder model
    The autoencoder model should be compiled using adam optimization and binary
        cross-entropy loss
    """
    inputs = keras.Input(shape=input_dims)
    enc = inputs

    for fil in filters:
        enc = keras.layers.Conv2D(fil,
                                  kernel_size=3,
                                  padding="same",
                                  activation="relu"
                                  )(enc)
        enc = keras.layers.MaxPooling2D(pool_size=2,
                                        padding="same"
                                        )(enc)

    dec_inputs = keras.Input((latent_dims))
    dec = dec_inputs

    last_filter = filters[-1]
    print(last_filter)
    filters = filters[:-1]
    print(filters)

    for fil in reversed(filters):
        dec = keras.layers.Conv2D(fil,
                                  kernel_size=3,
                                  padding="same",
                                  activation="relu"
                                  )(dec)
        dec = keras.layers.UpSampling2D(2)(dec)

    second_to_last = keras.layers.Conv2D(last_filter,
                                         kernel_size=3,
                                         padding="valid",
                                         activation="relu"
                                         )(dec)
    second_to_last = keras.layers.UpSampling2D(2)(second_to_last)
    outputs = keras.layers.Conv2D(input_dims[-1],
                                  kernel_size=3,
                                  padding="same",
                                  activation="sigmoid"
                                  )(second_to_last)

    encoder = keras.Model(inputs=inputs, outputs=enc)
    decoder = keras.Model(inputs=dec_inputs, outputs=outputs)

    auto_input = keras.Input(shape=input_dims)
    enc_in = encoder(auto_input)
    dec_out = decoder(enc_in)
    auto = keras.Model(inputs=auto_input, outputs=dec_out)

    loss = "binary_crossentropy"
    opt = keras.optimizers.Adam()

    encoder.compile(loss=loss, optimizer=opt)
    decoder.compile(loss=loss, optimizer=opt)
    auto.compile(loss=loss, optimizer=opt)

    return encoder, decoder, auto
