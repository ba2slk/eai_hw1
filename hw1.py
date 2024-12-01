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
    
def get_tag_info(train_dataset):
    # word별 tag 출현 횟수 Count
    tag_counts = defaultdict(list)  # '<word>' : [<tag>, <tag>]
    for sentence in train_dataset:
        for pair in sentence[1:-1]:
            tag_counts[pair[0]].append(pair[1])

    pos_counts = defaultdict(int)   
    for word, tags in tag_counts.items():
        tag_counts[word] = [Counter(tags), dict()]  # '<word>' : [Counter({<tag>: 20}, dict(): tag 빈도 계산])
        max_count = 0
        max_tag = None
        for tag in tag_counts[word][0]:
            pos_counts[tag] += tag_counts[word][0][tag]
            counts_per_tag = tag_counts[word][0][tag]  # 한 단어 내 특정 태그의 개수
            tag_counts[word][1][tag] = counts_per_tag
            if counts_per_tag > max_count:
                max_count = counts_per_tag
                max_tag = tag
        
        tag_counts[word][1]['max_count'] = max_count
        tag_counts[word][1]['max_tag'] = max_tag

    max_seen_tag = max(pos_counts, key=pos_counts.get) # 최다 출현 품사

    return tag_counts, max_seen_tag


def get_tagged_result(test_dataset, word_tag_infos, max_seen_tag):
    tagged_senteces_result = []
    for sentence in test_dataset:
        tagged_sentence = []
        for word in sentence:
            wt_pair = (None, None)
            if word == "START" or word == "END":
                wt_pair = (word, word)
            else:
                try: 
                    tag_info = word_tag_infos[word]
                    tag = tag_info[1]['max_tag']
                    wt_pair = (word, tag)
                except:
                    wt_pair = (word, max_seen_tag)
            tagged_sentence.append(wt_pair)
        tagged_senteces_result.append(tagged_sentence)

    return tagged_senteces_result




def baseline(train, test):
    '''
    Implementation for the baseline tagger.
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''

    # Get tag infos and max_seen_tag
    word_tag_infos, max_seen_tag = get_tag_info(train)

    # Get tagged_sentences_result
    tagged_sentences_result = get_tagged_result(test, word_tag_infos, max_seen_tag)
    
    return tagged_sentences_result

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