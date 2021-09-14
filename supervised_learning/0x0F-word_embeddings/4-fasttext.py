#!/usr/bin/env python3
"""
0x0F. Natural Language Processing - Word Embeddings
"""
from gensim.models import FastText


def fasttext_model(sentences, size=100, min_count=5, negative=5, window=5,
                   cbow=True, iterations=5, seed=0, workers=1):
    """
    Creates and trains a genism fastText model:

    sentences: a list of sentences to be trained on
    size is the dimensionality of the embedding layer
    min_count: the minimum number of occurrences of a word for use in training
    window: the maximum distance between the current and predicted word
        within a sentence
    negative: the size of negative sampling
    cbow: is a boolean to determine the training type; True is for CBOW;
        False is for Skip-gram
    iterations: is the number of iterations to train over
    seed: is the seed for the random number generator
    workers: is the number of worker threads to train the model
    Returns: the trained model
    """
    model = FastText(sentences=sentences, min_count=min_count,
                     iter=iterations, size=size,
                     window=window, sg=cbow,
                     seed=seed, negative=negative)

    model.train(sentences=sentences,
                total_examples=model.corpus_count,
                epochs=model.epochs)

    return model
