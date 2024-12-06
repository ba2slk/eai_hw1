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
    
'''------author: ba2slk------'''    
def get_dataset(train, test):
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
    
def get_tag_info(train_dataset, discard_se=True):
    # word별 tag 출현 횟수 Count
    tag_counts = defaultdict(list)  # '<word>' : [<tag>, <tag>]
    
    
    for sentence in train_dataset:
        if discard_se:
            for pair in sentence[1:-1]:
                tag_counts[pair[0]].append(pair[1])
        else:
            for pair in sentence:
                tag_counts[pair[0]].append(pair[1])

    pos_counts = defaultdict(int)   
    for word, tags in tag_counts.items():
        tag_counts[word] = [Counter([tag.strip() for tag in tags]), dict()]  # '<word>' : [Counter({<tag>: 20}, dict(): tag 빈도 계산])
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

    return tag_counts, max_seen_tag, pos_counts


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
    # # Turn off before checking the performance with run.py
    # train, test = get_dataset(train, test)

    # Get tag infos and max_seen_tag
    word_tag_infos, max_seen_tag, _ = get_tag_info(train, discard_se=True)

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

    # Turn off before checking the performance with run.py
    train, test = get_dataset(train, test)

    word_tag_infos, _, pos_counts = get_tag_info(train, discard_se=False)


    # 특정 태그가 특정 단어를 생성할 확률: Emission Probability
    emission_probability = defaultdict(lambda: defaultdict(int))
    for word, meta in word_tag_infos.items():
        for tag in meta[0]:
            word_counts_after_tag = meta[0][tag]
            total_counts_of_a_tag = pos_counts[tag]
            emission_probability[tag][word] = word_counts_after_tag / total_counts_of_a_tag # {<tag>: {<word>: <prob>, <word>: <prob>, ...}, ...}

    # 특정 태그 뒤에 특정 태그가 올 확률: Transition Probability
    transition_probability = defaultdict(lambda: defaultdict(int))
    for sentence in train:
        for i in range(len(sentence) - 1):
            current_tag = sentence[i][1]
            next_tag = sentence[i+1][1]
            transition_probability[current_tag][next_tag] += 1
    
    for tag, next_tags in transition_probability.items():
        total_next_tags = 0
        for _, counts in next_tags.items():
            total_next_tags += counts
                        
        for next_tag in next_tags.items():
            transition_probability[tag][next_tag[0]] = next_tag[1] / total_next_tags
    
    for k, v in emission_probability.items():
        print(k, v)

    for k, v in transition_probability.items():
        print(k, v)
            

    return 0       



def viterbi_ec(train, test):

    '''
    Implementation for the viterbi tagger.
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''

    raise NotImplementedError('viterbi_ec: not implemented')


if __name__ == "__main__":
    # For Personal Test
    train = "./data/brown-training.txt"
    test = "./data/brown-test.txt"

    # result = baseline(train, test)
    result = viterbi(train, test)
    # result = viterbi_ec(train, test)

    # print(result)