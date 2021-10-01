#!/usr/bin/env python3
"""
0x12 Transformer applications
"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """ Class that loads and preps a dataset for machine translation """

    def __init__(self, batch_size, max_len):
        """
        Creates the instance attributes:
        data_train, which contains the ted_hrlr_translate/pt_to_en
            tf.data.Dataset train split, loaded as_supervided
        data_valid, which contains the ted_hrlr_translate/pt_to_en
            tf.data.Dataset validate split, loaded as_supervided
        tokenizer_pt is the Portuguese tokenizer created from the training set
        tokenizer_en is the English tokenizer created from the training set
        """

        def _filter(x, y, max_length=max_len):
            """ filter """
            return tf.logical_and(tf.size(x) <= max_length,
                                  tf.size(y) <= max_length)

        examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                                       with_info=True,
                                       as_supervised=True)
        data_train, data_valid = examples['train'], examples['validation']
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            data_train)

        data_train = data_train.map(self.tf_encode)
        data_train = data_train.filter(_filter)
        data_train = data_train.cache()
        buffer_size = metadata.splits['train'].num_examples
        data_train = data_train.shuffle(buffer_size). \
            padded_batch(batch_size, padded_shapes=([None], [None]))
        self.data_train = data_train.prefetch(tf.data.experimental.AUTOTUNE)

        data_valid = data_valid.map(self.tf_encode)
        data_valid = data_valid.filter(_filter). \
            padded_batch(batch_size, padded_shapes=([None], [None]))
        self.data_valid = data_valid

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for our dataset:
        data:
            tf.data.Dataset whose examples are formatted as a tuple (pt, en)
            pt: tf.Tensor containing the Portuguese sentence
            en: tf.Tensor containing the corresponding English sentence
        The maximum vocab size should be set to 2**15
        Returns: tokenizer_pt, tokenizer_en
        tokenizer_pt is the Portuguese tokenizer
        tokenizer_en is the English tokenizer
        """
        tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
                    (en.numpy() for pt, en in data), target_vocab_size=2**15)

        tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
                    (pt.numpy() for pt, en in data), target_vocab_size=2**15)

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        Encodes a translation into tokens:
        pt is the tf.Tensor containing the Portuguese sentence
        en is the tf.Tensor containing the corresponding English sentence
        Tokenized sentences should include the start and end of sentence tokens
        The start token should be indexed as vocab_size
        The end token should be indexed as vocab_size + 1
        Returns: pt_tokens, en_tokens
            pt_tokens is a np.ndarray containing the Portuguese tokens
            en_tokens is a np.ndarray. containing the English tokens
        """
        pt_tokens = [self.tokenizer_pt.vocab_size] + self.tokenizer_pt.encode(
                    pt.numpy()) + [self.tokenizer_pt.vocab_size+1]

        en_tokens = [self.tokenizer_en.vocab_size] + self.tokenizer_en.encode(
            en.numpy()) + [self.tokenizer_en.vocab_size+1]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        Acts as a tensorflow wrapper for the encode instance method.
        """
        pt_answer, en_answer = tf.py_function(self.encode, [pt, en],
                                              [tf.int64, tf.int64])
        pt_answer.set_shape([None])
        en_answer.set_shape([None])

        return pt_answer, en_answer
