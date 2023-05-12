import readFasta
import numpy as np
def pssmfeature(fastas):
    pssmDir = r'../dbsnoPSSM'

    ans = []
    for i in fastas:
        encodings = []
        ad = []
        name, sequence = i[0], i[1]
        with open(pssmDir + '/' + name + '.pssm') as f:
            records = f.readlines()[3: 34]

        for i in sequence:
            if i == '-':
                ad.extend(i)
        add = [float(0)] * 20 * len(ad)

        code = []

        for line in records:
            xx_list = line.strip().split(" ")
            yy_list = [x for x in xx_list if x != ''][2:-2]
            zz_list = [float(x) for x in yy_list][0:20]
            encodings.extend(zz_list)

        ans.append(encodings)
        code.append(add)

    pssm = np.array(ans)
    return pssm
def pssmfeature1(fastas):
    pssmDir = r'../RecPSSM'
    ans = []

    index = 0
    for i in fastas:
        encodings = []
        ad = []
        code = []
        name, sequence = i[0], i[1]
        with open(pssmDir + '/' + name + '.pssm') as f:
            records = f.readlines()[3: 44]

        for line in records:
            xx_list = line.strip().split(" ")
            yy_list = [x for x in xx_list if x != ''][2:-2]
            zz_list = [float(x) for x in yy_list][0:20]
            encodings.extend(zz_list)
        for j in sequence:
            if j == '-':
                ad.extend(j)
        add = [float(0)] * 20 * len(ad)
        ans.append(encodings)
        code.append(add)
        for encodings in code:
            ans[index].extend(encodings)
        index += 1
    A = np.array(ans)
    return A
def get_data(pos, neg):
    from sklearn.model_selection import train_test_split
    train_p_data = pssmfeature1(pos)
    train_n_data = pssmfeature1(neg)

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
