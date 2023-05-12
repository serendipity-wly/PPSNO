import readFasta
import numpy as np

def generateGroupPairs(groupKey):
    gPair = {}
    for key1 in groupKey:
        for key2 in groupKey:
            gPair[key1+'.'+key2] = 0
    return gPair
def CKSAAGPfeature(fastas,gap=5):
    if gap < 0:
        print('Error: the gap should be equal or greater than zero' + '\n\n')
        return 0
    group = {
        'alphaticr': 'GAVLMI',
        'aromatic': 'FYW',
        'postivecharger': 'KRH',
        'negativecharger': 'DE',
        'uncharger': 'STCPNQ'
    }

    AA = 'ARNDCQEGHILKMFPSTWYV'

    groupKey = group.keys()

    index = {}
    for key in groupKey:
        for aa in group[key]:
            index[aa] = key

    gPairIndex = []
    for key1 in groupKey:
        for key2 in groupKey:
            gPairIndex.append(key1 + '.' + key2)

    encodings6 = []

    for i in fastas:
        sequence = i[1]
        code6 = []
        for g in range(gap + 1):
            gPair = generateGroupPairs(groupKey)
            sum5 = 0
            for p1 in range(len(sequence)):
                p2 = p1 + g + 1
                if p2 < len(sequence) and sequence[p1] in AA and sequence[p2] in AA:
                    gPair[index[sequence[p1]] + '.' + index[sequence[p2]]] = gPair[index[sequence[p1]] + '.' + index[
                        sequence[p2]]] + 1
                    sum5 = sum5 + 1

            if sum == 0:
                for gp in gPairIndex:
                    code6.append(0)
            else:
                for gp in gPairIndex:
                    code6.append(gPair[gp] / sum5)

        encodings6.append(code6)
    E = np.array(encodings6)
    return E
def get_data(pos, neg):
    from sklearn.model_selection import train_test_split
    train_p_data = CKSAAGPfeature(pos)
    train_n_data = CKSAAGPfeature(neg)

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