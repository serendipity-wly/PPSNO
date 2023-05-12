import readFasta
import numpy as np
from collections import Counter
import sys, os,re,platform,math
def QSOrderfeature(fastas, w=0.05,nlag = 30):
    dataFile = re.sub('codes$', '', os.path.split(os.path.realpath(__file__))[
        0]) + r'\data\Schneider-Wrede.txt' if platform.system() == 'Windows' else re.sub('codes$', '', os.path.split(
        os.path.realpath(__file__))[0]) + '/data/Schneider-Wrede.txt'
    dataFile1 = re.sub('codes$', '', os.path.split(os.path.realpath(__file__))[
        0]) + r'\data\Grantham.txt' if platform.system() == 'Windows' else re.sub('codes$', '', os.path.split(
        os.path.realpath(__file__))[0]) + '/data/Grantham.txt'

    AA = 'ACDEFGHIKLMNPQRSTVWY'
    AA1 = 'ARNDCQEGHILKMFPSTWYV'

    DictAA = {}
    for i in range(len(AA)):
        DictAA[AA[i]] = i

    DictAA1 = {}
    for i in range(len(AA1)):
        DictAA1[AA1[i]] = i

    with open(dataFile) as f:
        records = f.readlines()[1:]
    AADistance = []
    for i in records:
        array = i.rstrip().split()[1:] if i.rstrip() != '' else None
        AADistance.append(array)
    AADistance = np.array(
        [float(AADistance[i][j]) for i in range(len(AADistance)) for j in range(len(AADistance[i]))]).reshape((20, 20))

    with open(dataFile1) as f:
        records = f.readlines()[1:]
    AADistance1 = []
    for i in records:
        array = i.rstrip().split()[1:] if i.rstrip() != '' else None
        AADistance1.append(array)
    AADistance1 = np.array(
        [float(AADistance1[i][j]) for i in range(len(AADistance1)) for j in range(len(AADistance1[i]))]).reshape(
        (20, 20))

    encodings4 = []

    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        code4 = []
        arraySW = []
        arrayGM = []
        for n in range(1, nlag + 1):
            arraySW.append(
                sum([AADistance[DictAA[sequence[j]]][DictAA[sequence[j + n]]] ** 2 for j in range(len(sequence) - n)]))
            arrayGM.append(sum(
                [AADistance1[DictAA1[sequence[j]]][DictAA1[sequence[j + n]]] ** 2 for j in range(len(sequence) - n)]))
        myDict = {}
        for aa in AA1:
            myDict[aa] = sequence.count(aa)
        for aa in AA1:
            code4.append(myDict[aa] / (1 + w * sum(arraySW)))
        for aa in AA1:
            code4.append(myDict[aa] / (1 + w * sum(arrayGM)))
        for num in arraySW:
            code4.append((w * num) / (1 + w * sum(arraySW)))
        for num in arrayGM:
            code4.append((w * num) / (1 + w * sum(arrayGM)))
        encodings4.append(code4)
    D = np.array(encodings4)
    return D
def get_data(pos, neg):
    from sklearn.model_selection import train_test_split
    train_p_data = QSOrderfeature(pos)
    train_n_data = QSOrderfeature(neg)

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