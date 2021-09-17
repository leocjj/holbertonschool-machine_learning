#!/usr/bin/env python3
"""
0x10. Natural Language Processing - Evaluation Metrics
"""
import numpy as np


def uni_bleu(references, sentence):
    """
    Calculates the unigram BLEU score for a sentence:

    references is a list of reference translations
    each reference translation is a list of the words in the translation
    sentence is a list containing the model proposed sentence
    Returns: the unigram BLEU score
    """

    len_sentence = len(sentence)
    dict_words = {}

    for i in references:
        for word in i:
            if word in sentence and not dict_words.keys() == word:
                dict_words[word] = 1

    prob = sum(dict_words.values())
    ind = np.argmin([abs(len(x) - len_sentence) for x in references])
    best_match = len(references[ind])

    if len_sentence > best_match:
        bp = 1
    else:
        bp = np.exp(1 - float(best_match) / float(len_sentence))

    return bp * np.exp(np.log(prob / len_sentence))
