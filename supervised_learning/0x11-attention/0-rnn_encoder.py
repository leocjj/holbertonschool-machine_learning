#!/usr/bin/env python3
"""
0x11. Attention
"""
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """
    class to encode for machine translation
    """
    def __init__(self, vocab, embedding, units, batch):
        """
        vocab is an integer representing the size of the input vocabulary
        embedding: integer representing dimensionality of theembedding vector
        units: integer representing the number of hidden units in the RNN cell
        batch: is an integer representing the batch size
        Sets: the following public instance attributes:
            batch - the batch size
            units - the number of hidden units in the RNN cell
            embedding - a keras Embedding layer that converts words from the
                vocabulary into an embedding vector
            gru - a keras GRU layer with units units
                Should return both the full sequence of outputs as well as the
                    last hidden state
                Recurrent weights should be initialized with glorot_uniform
        """
        super().__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(self.units,
                                       recurrent_initializer='glorot_uniform',
                                       return_sequences=True,
                                       return_state=True)

    def initialize_hidden_state(self):
        """
        Public instance method def initialize_hidden_state(self):
        Initializes the hidden states for the RNN cell to a tensor of zeros
        Returns: a tensor of shape (batch, units)containing the initialized
            hidden states
        """

        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """
        Public instance method def call(self, x, initial):
        x is a tensor of shape (batch, input_seq_len) containing the input to
            the encoder layer as word indices within the vocabulary
        initial: tensor of shape (batch, units) containing initial hidden state
        Returns: outputs, hidden
        outputs is a tensor of shape (batch, input_seq_len, units)containing
            the outputs of the encoder
        hidden is a tensor of shape (batch, units) containing the last hidden
            state of the encoder
        """
        x = self.embedding(x)
        outputs, hidden = self.gru(x, initial_state=initial)

        return outputs, hidden
