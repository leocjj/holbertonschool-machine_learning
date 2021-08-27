#!/usr/bin/env python3
"""
0x04. Autoencoders
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates an autoencoder:
    input_dims is an integer containing the dimensions of the model input
    hidden_layers is a list containing the number of nodes for each hidden
        layer in the encoder, respectively
    the hidden layers should be reversed for the decoder
    latent_dims is an integer containing the dimensions of the latent space
        representation
    Returns: encoder, decoder, auto
        encoder is the encoder model
        decoder is the decoder model
        auto is the full autoencoder model
    The autoencoder model should be compiled using adam optimization and binary
        cross-entropy loss
    All layers should use a relu activation except for the last layer in the
        decoder, which should use sigmoid
    """

    input_layer = keras.Input(shape=(input_dims,))
    enc = input_layer

    for layer_dims in hidden_layers:
        enc = keras.layers.Dense(units=layer_dims, activation="relu")(enc)

    bot_neck = keras.layers.Dense(units=latent_dims, activation="relu")(enc)

    dec_input = keras.Input(shape=(latent_dims,))
    dec = dec_input

    for layer_dims in reversed(hidden_layers):
        dec = keras.layers.Dense(units=layer_dims, activation="relu")(dec)

    out_put = keras.layers.Dense(units=input_dims, activation="sigmoid")(dec)

    encoder = keras.Model(inputs=input_layer, outputs=bot_neck)
    decoder = keras.Model(inputs=dec_input, outputs=out_put)

    auto_input = keras.Input(shape=(input_dims,))
    enc_in = encoder(auto_input)
    dec_out = decoder(enc_in)
    auto = keras.Model(inputs=auto_input, outputs=dec_out)

    loss = "binary_crossentropy"
    opt = keras.optimizers.Adam()

    encoder.compile(loss=loss, optimizer=opt)
    decoder.compile(loss=loss, optimizer=opt)
    auto.compile(loss=loss, optimizer=opt)

    return encoder, decoder, auto
