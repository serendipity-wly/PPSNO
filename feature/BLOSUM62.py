import readFasta
import numpy as np
def BLOSUM62feature(fastas):
    blosum62 = {
        'A': [-1.56, -1.67, -0.97, -0.27, -0.93, -0.78, -0.2, -0.08, 0.21, -0.48],  # A
        'R': [0.22, 1.27, 1.37, 1.87, -1.7, 0.46, 0.92, -0.39, 0.23, 0.93],  # R
        'N': [1.14, -0.07, -0.12, 0.81, 0.18, 0.37, -0.09, 1.23, 1.1, -1.73],  # N
        'D': [0.58, -0.22, -1.58, 0.81, -0.92, 0.15, -1.52, 0.47, 0.76, 0.7],  # D
        'C': [0.12, -0.89, 0.45, -1.05, -0.71, 2.41, 1.52, -0.69, 1.13, 1.1],  # C
        'Q': [-0.47, 0.24, 0.07, 1.1, 1.1, 0.59, 0.84, -0.71, -0.03, -2.33],  # Q
        'E': [-1.45, 0.19, -1.61, 1.17, -1.31, 0.4, 0.04, 0.38, -0.35, -0.12],  # E
        'G': [1.46, -1.96, -0.23, -0.16, 0.1, -0.11, 1.32, 2.36, -1.66, 0.46],  # G
        'H': [-0.41, 0.52, -0.28, 0.28, 1.61, 1.01, -1.85, 0.47, 1.13, 1.63],  # H
        'I': [-0.73, -0.16, 1.79, -0.77, -0.54, 0.03, -0.83, 0.51, 0.66, -1.78],  # I
        'L': [-1.04, 0, -0.24, -1.1, -0.55, -2.05, 0.96, -0.76, 0.45, 0.93],  # L
        'K': [-0.34, 0.82, -0.23, 1.7, 1.54, -1.62, 1.15, -0.08, -0.48, 0.6],  # K
        'M': [-1.4, 0.18, -0.42, -0.73, 2, 1.52, 0.26, 0.11, -1.27, 0.27],  # M
        'F': [-0.21, 0.98, -0.36, -1.43, 0.22, -0.81, 0.67, 1.1, 1.71, -0.44],  # F
        'P': [2.06, -0.33, -1.15, -0.75, 0.88, -0.45, 0.3, -2.3, 0.74, -0.28],  # P
        'S': [0.81, -1.08, 0.16, 0.42, -0.21, -0.43, -1.89, -1.15, -0.97, -0.23],  # S
        'T': [0.26, -0.7, 1.21, 0.63, -0.1, 0.21, 0.24, -1.15, -0.56, 0.19],  # T
        'W': [0.3, 2.1, -0.72, -1.57, -1.16, 0.57, -0.48, -0.4, -2.3, -0.6],  # W
        'Y': [1.38, 1.48, 0.8, -0.56, 0, -0.68, -0.31, 1.03, -0.05, 0.53],  # Y
        'V': [-0.74, -0.71, 2.04, -0.4, 0.5, -0.81, -1.07, 0.06, -0.46, 0.65],  # V
        '-': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # -
    }
    encodings = []
    for i in fastas:
        name, sequence = i[0], i[1]
        code = []
        for aa in sequence:
            code = code + blosum62[aa]
        encodings.append(code)
    B = np.array(encodings)
    return B
def get_data(pos, neg):
    from sklearn.model_selection import train_test_split
    train_p_data = BLOSUM62feature(pos)
    train_n_data = BLOSUM62feature(neg)

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