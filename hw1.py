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
    
def get_dataset(train, test):
    """ Parse Datasets"""

    train_dataset = []
    with open(train, 'r') as t:
        for sentence in t:
            pairs = [("START", "START")]
            for pair in sentence.split(' '):
                splitted = pair.split("=")
                pair_tup = (splitted[0], splitted[1])
                pairs.append(pair_tup)
            pairs.append(("END", "END"))
            train_dataset.append(pairs)
    # print(train_sentences[0])

    test_dataset = []
    with open(test, 'r') as t:
        for sentence in t:
            words = ["START"]
            for word in sentence.split(' '):
                splitted = word.split("=")
                words.append(splitted[0])
            words.append("END")
            test_dataset.append(words)
    #print(test_dataset)

    return train_dataset, test_dataset

def get_tag_info(train_dataset):
    # word별 tag 빈도 Count하기
    tag_counts = defaultdict(list)  # '<word>' : [<tag>, <tag>]
    for sentence in train_dataset:
        for pair in sentence[1:-1]:
            tag_counts[pair[0]].append(pair[1])
    
    
    total_word_counts = 0  # 모든 단어 빈도
    for word, tags in tag_counts.items():
        tag_counts[word] = [Counter(tags), dict()]  # '<word>' : [Counter({<tag>: 20}, dict(): tag 빈도 계산])
        word_freq = 0  # 문서 내 해당 단어의 빈도 == sum of tags per word
        for tag in tag_counts[word][0]:
            counts_per_tag = tag_counts[word][0][tag]  # 한 단어 내 특정 태그의 개수
            word_freq += counts_per_tag
        total_word_counts += word_freq  # 전체 단어 수 합산

        max_prob = 0
        max_tag = None
        for tag in tag_counts[word][0]:
            counts_per_tag = tag_counts[word][0][tag]  # 한 단어 내 특정 태그의 개수
            tag_proportion_in_a_word = counts_per_tag / word_freq
            tag_counts[word][1][tag] = tag_proportion_in_a_word
            if tag_proportion_in_a_word > max_prob:
                max_prob = tag_proportion_in_a_word
                max_tag = tag
        
        tag_counts[word][1]['max_prob'] = max_prob
        tag_counts[word][1]['max_tag'] = max_tag
        tag_counts[word][1]['word_freq'] = word_freq

    # Unseen Word에 대한
    max_seen_word_count = 0 # 가장 많이 출현한 단어의 출현 횟수 (int)
    for word, meta in tag_counts.items():
        max_seen_word_count = max(max_seen_word_count, meta[1]['word_freq'])
    max_seen_probability = max_seen_word_count / total_word_counts 

    return tag_counts, max_seen_probability




def baseline(train, test):
    '''
    Implementation for the baseline tagger.
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    
    train_dataset, test_dataset = get_dataset(train, test)
    word_tag_info, max_seen_probability = get_tag_info(train_dataset)

    print(max_seen_probability)

    print(word_tag_info['unconscious'])

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

if __name__ == "__main__":
        baseline('./data/brown-training.txt', './data/brown-test.txt')