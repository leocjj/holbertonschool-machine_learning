#!/usr/bin/env python3
"""
0x10. Natural Language Processing - Evaluation Metrics
"""
import numpy as np


def counter(phrase, n):
    """Return dict with count of words"""
    tuple_list = [(tuple(i for i in phrase[x:x + n]))
                  for x in range(len(phrase) - n + 1)]
    d = {}
    for x in range(len(phrase) - n + 1):
        key = tuple(i for i in phrase[x:x + n])
        if key not in d:
            d[key] = tuple_list.count(key)

    return d


def count_clip(references, sentence, n):
    """Count clip"""
    res = {}
    ct_sentence = counter(sentence, n)
    for ref in references:
        ct_ref = counter(ref, n)
        for k in ct_ref:
            if k in res:
                res[k] = max(ct_ref[k], res[k])
            else:
                res[k] = ct_ref[k]
    count = {k: min(ct_sentence.get(k, 0), res.get(k, 0)) for k in ct_sentence}

    return count


def modified_precision(references, sentence, n):
    """Modified precision"""
    ct_clip = count_clip(references, sentence, n)
    ct = counter(sentence, n)

    return sum(ct_clip.values()) / float(max(sum(ct.values()), 1))


def ngram_bleu(references, sentence, n):
    """
    Calculates the n-gram BLEU score for a sentence:

    references is a list of reference translations
    each reference translation is a list of the words in the translation
    sentence is a list containing the model proposed sentence
    n is the size of the n-gram to use for evaluation
    Returns: the n-gram BLEU score
    """
    W = [0.25] * 4
    Pn = [modified_precision(references, sentence, n)
          for ngram, _ in enumerate(W, start=1)]
    c = len(sentence)
    closest_ref_idx = np.argmin([abs(len(x) - c) for x in references])
    r = len(references[closest_ref_idx])
    if c > r:
        BP = 1
    else:
        BP = np.exp(1 - (float(r) / c))
    score = np.sum([(wn * np.log(Pn[i])) if Pn[i] != 0 else 0
                    for i, wn in enumerate(W)])

    return BP * np.exp(score)
