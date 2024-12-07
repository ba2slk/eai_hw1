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

def safe_log (probability):
    return np.log(probability + EPSILON)

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
    # train, test = get_dataset(train, test)

    word_tag_infos, _, pos_counts = get_tag_info(train, discard_se=False)
    # print(pos_counts)

    # for k, v in word_tag_infos.items():
    #     print(k, v)
    # for k, v in pos_counts.items():
    #     print(k, v)
    

    # 특정 태그가 특정 단어를 생성할 확률: Emission Probability
    emission_probability = defaultdict(lambda: defaultdict(float))
    for word, meta in word_tag_infos.items():
        for tag in meta[0]:
            word_counts_after_tag = meta[0][tag]
            total_counts_of_a_tag = pos_counts[tag]
            emission_probability[tag][word] = word_counts_after_tag / total_counts_of_a_tag # {<tag>: {<word>: <prob>, <word>: <prob>, ...}, ...}

    # 특정 태그 뒤에 특정 태그가 올 확률: Transition Probability
    transition_probability = defaultdict(lambda: defaultdict(float))
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

    # 초기 확률 분포 -> START 다음에 나오는 애들 확률로
    initial_probability_distribution = transition_probability['START']
    tag_state_space = list(pos_counts.keys())
    # tag_state_space = [tag for tag in pos_counts.keys() if tag not in ["START", "END"]]

    # for k, v in emission_probability.items():
    #     print(k, v)

    # for k, v in transition_probability.items():
    #     print(k, v)

    tagged_sentences_result = []
    for sentence in test:
        #sentence = test[0]
        tagged_sentence = [("END", "END")]
        # V = defaultdict(lambda: defaultdict(float))  # Viterbi Matrix -> 각 단계에서의 최대 확률 저장
        # BT = defaultdict(lambda: defaultdict(str))

        T = len(sentence) - 1
        V = np.full((T, len(tag_state_space)), -np.inf)
        BT = np.zeros((T, len(tag_state_space))) # 꼭 수정하기

        prev_word_idx = 0
        for i in range(0, T):
            curr_word_idx = i
            for curr_tag_idx in range(len(tag_state_space)):
                curr_tag = tag_state_space[curr_tag_idx]
                curr_word = sentence[curr_word_idx]
                ep = np.log(emission_probability[curr_tag][curr_word] + EPSILON)
                if i == 0:
                    ip = initial_probability_distribution[curr_tag]
                    V[curr_word_idx][curr_tag_idx] = np.log(ip + EPSILON) + ep
                    BT[curr_word_idx][curr_tag_idx] = curr_tag_idx
                else:
                    max_prob = -np.inf
                    max_prev_tag_idx = -1
                    for prev_tag_idx in range(len(tag_state_space)):
                        prev_tag = tag_state_space[prev_tag_idx]
                        tp = transition_probability[prev_tag][curr_tag]
                        candidate_prob = V[prev_word_idx][prev_tag_idx] + np.log(tp + EPSILON) + ep
                        if candidate_prob > max_prob:
                            max_prob = candidate_prob
                            max_prev_tag_idx = prev_tag_idx
                    V[curr_word_idx][curr_tag_idx] = max_prob
                    BT[curr_word_idx][curr_tag_idx] = max_prev_tag_idx
            prev_word_idx = i

        last_word_idx = prev_word_idx
        max_prob = np.max(V[last_word_idx])
        last_word_tag_idx = np.argmax(V[last_word_idx])
        # tagged_sentence.insert(1, (sentence[last_word_idx], tag_state_space[last_word_tag_idx]))
        tagged_sentence.append((sentence[last_word_idx], tag_state_space[last_word_tag_idx]))
        prev_tag_idx = int(BT[last_word_idx][last_word_tag_idx])

        for curr_word_idx in range (T-2, 0, -1):
            curr_word = sentence[curr_word_idx]
            # tagged_sentence.insert(1, (curr_word, tag_state_space[prev_tag_idx]))
            tagged_sentence.append((curr_word, tag_state_space[prev_tag_idx]))
            prev_tag_idx = int(BT[curr_word_idx][prev_tag_idx])

        tagged_sentence.append(("START", "START"))
        tagged_sentence.reverse()
        tagged_sentences_result.append(tagged_sentence)
        
    return tagged_sentences_result


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