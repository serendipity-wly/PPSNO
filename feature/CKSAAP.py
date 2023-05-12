import readFasta
import numpy as np
import re,math
def CKSAAPfeature(fastas, gap=5):
    if gap < 0:
        print('Error: the gap should be equal or greater than zero' + '\n\n')
        return 0
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    encodings8 = []
    aaPairs1 = []
    for aa1 in AA:
        for aa2 in AA:
            aaPairs1.append(aa1 + aa2)
    for i in fastas:
        name, sequence = i[0], i[1]
        code8 = []
        for g in range(gap + 1):
            myDict = {}
            for pair in aaPairs1:
                myDict[pair] = 0
            sum4 = 0
            for index1 in range(len(sequence)):
                index2 = index1 + g + 1
                if index1 < len(sequence) and index2 < len(sequence) and sequence[index1] in AA and sequence[
                    index2] in AA:
                    myDict[sequence[index1] + sequence[index2]] = myDict[sequence[index1] + sequence[index2]] + 1
                    sum4 = sum4 + 1
            for pair in aaPairs1:
                code8.append(myDict[pair] / sum4)
        encodings8.append(code8)
    F = np.array(encodings8)
    return F
def get_data(pos, neg):
    from sklearn.model_selection import train_test_split
    train_p_data = CKSAAPfeature(pos)
    train_n_data = CKSAAPfeature(neg)

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