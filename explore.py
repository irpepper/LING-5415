import os
import re

import sys
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from utils.embeddings import load_glove

GLOVE_DIR = "../glove"
TRAIN_DATA = "./data/train.csv"
TEST_DATA = "./data/test.csv"

embedding_matrix, nb_words = load_glove()

#Load embeddings
embeddings_index = {}
with open(os.path.join(GLOVE_DIR, 'glove.6B.50d.txt'), encoding="utf8") as f:
    for line in f:
        values = line.split(" ")
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs


# def clean_special_chars(text):
#     punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
#     mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', }
#     text = re.sub(r'(\\\\){0,2}x[\d\w]{2}','',text)
#     for p in mapping:
#         text = text.replace(p, mapping[p])
#
#     for p in punct:
#         text = text.replace(p, " "+p+" ")
#
#     specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}  # Other special characters that I have to deal with in last
#     for s in specials:
#         text = text.replace(s, specials[s])
#     return text

def replace_usernames(text):
    return re.sub(r'@[a-zA-Z0-9_]{1,15}','$USER',text)

def replace_hashtags(text):
    return re.sub(r'#[a-zA-Z0-9_]*','$HASHTAG',text)

def replace_urls(text):
    return re.sub(r'[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)','$URL',text)

def clean_text(text):
    text = replace_usernames(text)
    text = replace_hashtags(text)
    text = replace_urls(text)
    # text = clean_special_chars(text)
    return text

df_train= pd.read_csv(TRAIN_DATA)
df_test= pd.read_csv(TEST_DATA)
df_train["Tweet"] = df_train["Tweet"].apply(lambda x: clean_text(x))
df_test["Tweet"] = df_test["Tweet"].apply(lambda x: clean_text(x))
texts = df_train["Tweet"]
# texts = df_test["Tweet"]

tokenizer = Tokenizer(num_words=None)

tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index


oov_count = 0
oov = []
for word in word_index:
    if word not in embeddings_index.keys():
        oov_count += 1
        if word not in oov:
            oov.append(word)

oov
print("Number OOV words:{}\nNumber of words in vocab:{}\nVocab Coverage:{}".format(oov_count,len(word_index),(len(word_index) - oov_count)/len(word_index)))



from collections import Counter
tokenized_texts = texts.apply(text_to_word_sequence)
tokenized_texts = list(tokenized_texts)
tokenized_texts
X = Counter([y for x in tokenized_texts for y in x])
X.most_common()

drugs = [r'dioralyte', r'solpadine', r'solpadol',
         r'icyhot', r'zanaflex', r'tramdol',
         r'moldavite', r'robex', r'novacaine',
         r'tramadol', r'pedialite', r'NeoCitran',
         r'dydramol', r'adderal', r'ciroc', r'im+odium',
         r'profen', r'excedrine', r'zanny', r'gaviscon',
         r'cod[ea]mol', r'panadol', r'ora[jg]el', r'ben[ea]dryl',
         r'percocet', r'mucinex', r'dayquil',
         r'nyquil', r'alieve', r'sudocrem',
         r'hydroco', r'propionate', r'fluticasone',
         r'liefern', r'medrol', r'klonopin',
         r'proair', r'arcoxia', r'spiriva',
         r'anacin', r'accolate', r'advair',
         r'alprazolam', r'lorazepam',
         r'symbicort', r'acetaminophen', r'methadone',
         r'oxyc', r'robitussin', r'fluoxetine',
         r'diazi?e?pam', r'advil', r'tylenol',
         r'steroid', r'ibuprofen', r'motrin',
         r'valium', r'panadol', r'code?ine',
         r'flonase', r'ativan', r'proza[ck]',
         r'vent[oia]lin', r'pred(ni|in)sone',
         r'xan([enya]x|s)', r'paracet(am|ma)ol',
         r'albuter[oa]l', r'clona[sz]epam']

failed = []
for word in X.most_common():
    flag = False
    for drug in drugs:
        if re.search(drug,word[0],re.I):
            flag = True
    if not flag:
        failed.append(word[0])
failed

failed = []
for tweet in texts:
    flag = False
    for drug in drugs:
        if re.search(drug,tweet,re.I):
            flag = True
    if not flag:
        failed.append(tweet)
len(failed)
failed
