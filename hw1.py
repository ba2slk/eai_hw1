from collections import defaultdict, Counter
from math import log
import numpy as np

EPSILON = 1e-5

def smoothed_prob(arr, alpha=1):
    '''
    list of probabilities smoothed by Laplace smoothing
    input: arr (list or numpy.ndarray of integers which are counts of any elements)
           alpha (Laplace smoothing parameter. No smoothing if zero)
    output: list of smoothed probabilities

    E.g., smoothed_prob( arr=[0, 1, 3, 1, 0], alpha=1 ) -> [0.1, 0.2, 0.4, 0.2, 0.1]
          smoothed_prob( arr=[1, 2, 3, 4],    alpha=0 ) -> [0.1, 0.2, 0.3, 0.4]
    '''
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)

    _sum = arr.sum()
    if _sum:
        return ((arr + alpha) / (_sum + arr.size * alpha)).tolist()
    else:
        return ((arr + 1) / arr.size).tolist()

def baseline(train, test):
    '''
    Implementation for the baseline tagger.
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    raise NotImplementedError("You need to write this part!")


def viterbi(train, test):

    '''
    Implementation for the viterbi tagger.
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    raise NotImplementedError("You need to write this part!")

def viterbi_ec(train, test):

    '''
    Implementation for the viterbi tagger.
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    raise NotImplementedError("You need to write this part!")