import readFasta
import numpy as np
import re,math
def DDEfeature(fastas):
    AA= 'ACDEFGHIKLMNPQRSTVWY'

    myCodons = {
        'A': 4,
        'C': 2,
        'D': 2,
        'E': 2,
        'F': 2,
        'G': 4,
        'H': 2,
        'I': 3,
        'K': 2,
        'L': 6,
        'M': 1,
        'N': 2,
        'P': 4,
        'Q': 2,
        'R': 6,
        'S': 6,
        'T': 4,
        'V': 4,
        'W': 1,
        'Y': 2
    }

    encodings3 = []
    diPeptides = [aa1 + aa2 for aa1 in AA for aa2 in AA]

    myTM = []
    for pair in diPeptides:
        myTM.append((myCodons[pair[0]] / 61) * (myCodons[pair[1]] / 61))
    AADict10 = {}
    for i in range(len(AA)):
        AADict10[AA[i]] = i
    for i in fastas:
        sequence = re.sub('-', '', i[1])
        code3 = []
        tmpCode = [0] * 400
        for j in range(len(sequence) - 2 + 1):
            tmpCode[AADict10[sequence[j]] * 20 + AADict10[sequence[j + 1]]] = tmpCode[AADict10[sequence[j]] * 20 + AADict10[sequence[j + 1]]] + 1
        if sum(tmpCode) != 0:
            tmpCode = [i / sum(tmpCode) for i in tmpCode]

        myTV = []
        for j in range(len(myTM)):
            myTV.append(myTM[j] * (1 - myTM[j]) / (len(sequence) - 1))

        for j in range(len(tmpCode)):
            tmpCode[j] = (tmpCode[j] - myTM[j]) / math.sqrt(myTV[j])

        code3 = code3 + tmpCode
        encodings3.append(code3)
    C = np.array(encodings3)
    return C
def get_data(pos, neg):
    from sklearn.model_selection import train_test_split
    train_p_data = DDEfeature(pos)
    train_n_data = DDEfeature(neg)

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