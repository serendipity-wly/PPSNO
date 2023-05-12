import os
import numpy as np
from Bio import SeqIO
from sklearn.model_selection import train_test_split
from propy import PyPro
import propy
from sklearn.model_selection import train_test_split
from Bio import SeqIO
from transformers import AutoModel
import random
# from datasets import load_dataset
from transformers import BertTokenizer
from transformers import FeatureExtractionPipeline

import tensorflow.keras as keras

import torch
torch.set_num_threads(40)

from propy.PyPro import GetProDes

from propy.AAComposition import (
    CalculateAAComposition,
    CalculateAADipeptideComposition,
    CalculateDipeptideComposition,
    GetSpectrumDict,
)

std=["A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V"]
def read_data(path):
    res = {}
    rx = list(SeqIO.parse(path,format="fasta"))
    for x in rx:
        id = str(x.id)
        seq = str(x.seq).upper()
        seq = "".join([x for x in list(seq) if x in std])
        res[id]=seq
    return res


from transformers import T5Tokenizer, T5Model
import re

#pip install sentencepiece
tokenizer_T5 = T5Tokenizer.from_pretrained('/home/ys/SNO/T5/prot_t5_xl_uniref50', do_lower_case=False)
model_T5 = T5Model.from_pretrained("/home/ys/SNO/T5/prot_t5_xl_uniref50")
def get_t5xl_bert(seq):
    sequences_Example = [" ".join(list(seq))]
    ids = tokenizer_T5.batch_encode_plus(sequences_Example, add_special_tokens=True, padding=True)
    input_ids = torch.tensor(ids['input_ids'])
    attention_mask = torch.tensor(ids['attention_mask'])
    with torch.no_grad():
        embedding = model_T5(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=input_ids)
    encoder_embedding = embedding[2].cpu().numpy()
    decoder_embedding = embedding[0].cpu().numpy()

    res = encoder_embedding
    res = np.array(res)[0,1:, :]
    xx = np.zeros((31, 1024))
    tmp = res[0:31, :]
    xx[:tmp.shape[0], :]=tmp

    return xx


def encoding(seq):
    res = get_t5xl_bert(seq)
    return res


sno_path = "/home/ys/SNO/T5/Reviewed.txt"
nonsno_path = "/home/ys/SNO/T5/Reviewed.txt"

sno_seqs = list(read_data(sno_path).values())
nonsno_seqs = list(read_data(nonsno_path).values())

print("sno seq",len(sno_seqs))
print("nonsno seq",len(nonsno_seqs))

seed = 3
train_sno_seqs, test_sno_seqs = train_test_split(sno_seqs, test_size=0.4, random_state=seed)
val_sno_seqs, test_sno_seqs = train_test_split(test_sno_seqs, test_size=0.5, random_state=seed)

train_nonsno_seqs, test_nonsno_seqs = train_test_split(nonsno_seqs, test_size=0.4, random_state=seed)
val_nonsno_seqs, test_nonsno_seqs = train_test_split(test_nonsno_seqs, test_size=0.5, random_state=seed)


from multiprocessing import Pool,cpu_count
pool = Pool(cpu_count())
train_sno_data = np.array([encoding(seq) for seq in train_sno_seqs])
train_nonsno_data = np.array([encoding(seq) for seq in train_nonsno_seqs])
val_sno_data = np.array([encoding(seq) for seq in val_sno_seqs])
val_nonsno_data = np.array([encoding(seq) for seq in val_nonsno_seqs])
test_sno_data = np.array([encoding(seq) for seq in test_sno_seqs])
test_nonsno_data = np.array([encoding(seq) for seq in test_nonsno_seqs])

print("train sno",len(train_sno_seqs))
print("train nonsno",len(train_nonsno_seqs))

train_data = np.concatenate([train_sno_data, train_nonsno_data])
val_data = np.concatenate([val_sno_data, val_nonsno_data])
test_data = np.concatenate([test_sno_data, test_nonsno_data])

train_label = np.array([1] * len(train_sno_data) + [0] * len(train_nonsno_data))
val_label = np.array([1] * len(val_sno_data)  + [0] * len(val_nonsno_data))
test_label = np.array([1] * len(test_sno_data) + [0] * len(test_nonsno_data))

train_data = np.array(train_data)
val_data = np.array(val_data)
test_data = np.array(test_data)
train_label = np.array(train_label)
val_label = np.array(val_label)
test_label = np.array(test_label)
print('11',np.array(train_data).shape)
print('21',np.array(val_data).shape)
print('31',np.array(test_data).shape)
print('12',np.array(train_label).shape)
print('23',np.array(val_label).shape)
print('34',np.array(test_label).shape)

# np.save("train_data_trainhum_T5bert.npy", train_data)
# np.save("val_data_trainhum_T5bert.npy", val_data)
# np.save("test_data_trainhum_T5bert.npy", test_data)
# np.save("train_label_trainhum_T5bert.npy", train_label)
# np.save("val_label_trainhum_T5bert.npy", val_label)
# np.save("test_label_trainhum_T5bert.npy", test_label)
train_data = np.load("train_data_T5Reviewed_bert.npy")
val_data = np.load("val_data_T5Reviewed_bert.npy")
test_data = np.load("test_data_T5Reviewed_bert.npy")
train_label = np.load("train_label_T5Reviewed_bert.npy")
val_label = np.load("val_label_T5Reviewed_bert.npy")
test_label = np.load("test_label_T5Reviewed_bert.npy")
