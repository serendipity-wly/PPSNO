import readFasta
import numpy as np
from collections import Counter
import sys, os,re,platform,math
def Count(seq1, seq2):
    sum = 0
    for aa in seq1:
        sum = sum + seq2.count(aa)
    return sum
def EGAACfeature(fastas,window=5):
    if window < 1:
        print('Error: the sliding window should be greater than zero' + '\n\n')
        return 0

    group9 = {
        'alphaticr': 'GAVLMI',
        'aromatic': 'FYW',
        'postivecharger': 'KRH',
        'negativecharger': 'DE',
        'uncharger': 'STCPNQ'
    }

    groupKey = group9.keys()

    encodings9 = []

    for i in fastas:
        name, sequence = i[0], i[1]
        code9 = []
        for j in range(len(sequence)):
            if j + window <= len(sequence):
                count = Counter(sequence[j:j + window])
                myDict = {}
                for key in groupKey:
                    for aa in group9[key]:
                        myDict[key] = myDict.get(key, 0) + count[aa]
                for key in groupKey:
                    code9.append(myDict[key] / window)
        encodings9.append(code9)
    I = np.array(encodings9)
    return I
def get_data(pos, neg):
    from sklearn.model_selection import train_test_split
    train_p_data = EGAACfeature(pos)
    train_n_data = EGAACfeature(neg)

    train_p_data, test_p_data = train_test_split(train_p_data, test_size=0.4, random_state=3)
    val_p_data, test_p_data = train_test_split(test_p_data, test_size=0.5, random_state=3)
    train_n_data, test_n_data = train_test_split(train_n_data, test_size=0.4, random_state=3)
    val_n_data, test_n_data = train_test_split(test_n_data, test_size=0.5, random_state=3)
    train_p_data = np.array(train_p_data)
    train_n_data = np.array(train_n_data)
    val_p_data = np.array(val_p_data)
    val_n_data = np.array(val_n_data)
    train_data = np.concatenate([train_p_data, train_n_data], axis=0)
    val_data = np.concatenate([val_p_data, val_n_data], axis=0)
    test_data = np.concatenate([test_p_data, test_n_data], axis=0)
    train_label = [1] * len(train_p_data) + [0] * len(train_n_data)
    val_label = [1] * len(val_p_data) + [0] * len(val_n_data)
    test_label = [1] * len(test_p_data) + [0] * len(test_n_data)
    train_label = np.array(train_label)
    val_label = np.array(val_label)
    test_label = np.array(test_label)
    return train_data, val_data, test_data, train_label, val_label, test_label
fastas1 = readFasta.readFasta('../dataset/Rec_pos.txt')
fastas = readFasta.readFasta('../dataset/Rec_neg.txt')
train_data1, val_data1, test_data1, train_label, val_label, test_label = get_data(fastas1, fastas)
print(np.array(train_data1).shape)
print(np.array(val_data1).shape)
print(np.array(test_data1).shape)
print(np.array(train_label).shape)
print(np.array(val_label).shape)
print(np.array(test_label).shape)