#!/usr/bin/env python3
"""
0x0F. Natural Language Processing - Word Embeddings
"""
from gensim.models import Word2Vec


def gensim_to_keras(model):
    """
    Converts a gensim word2vec model to a keras Embedding layer:

    model is a trained gensim word2vec models
    Returns: the trainable keras Embedding
    """
    layer = model.wv.get_keras_embedding()

    return layer
