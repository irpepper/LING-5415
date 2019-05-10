import spacy
from spacy.tokens import Doc, Span, Token
from tqdm import tqdm
import time
import re
import pandas as pd
import numpy as np
import gensim
from nltk.stem import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import SnowballStemmer
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical

import random, os, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.initializers import *

from tensorflow.python.keras.layers import Layer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.backend as K
import sklearn.metrics as sklm


np.random.seed(1) # NumPy
import random
random.seed(2) # Python
from tensorflow.random import set_seed
set_seed(3) # Tensorflow

# preprocessing functions
def replace_usernames(text):
    return re.sub(r'@[a-zA-Z0-9_]{1,15}', '_USER_', text)


def replace_hashtags(text):
    return re.sub(r'#[a-zA-Z0-9_]*', '_HASHTAG_', text)


def replace_urls(text):
    pattern = r'[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)'
    pattern2 = r'https?:\/\/t.co\/[\w\d]*'
    text = re.sub(pattern, '_URL_', text)
    return re.sub(pattern2, '_URL_', text)

def replace_amps(text):
    return re.sub(r'&amp', 'and', text)

def clean_text(text):
    text = replace_usernames(text)
    text = replace_hashtags(text)
    text = replace_urls(text)
    text = replace_amps(text)
    return text


# Load spacy
start_time = time.time()
print("Loading Spacy Model")
nlp = spacy.load('en_core_web_lg')
print("{} seconds".format(time.time() - start_time))

# Custom spacy attributes
Doc.set_extension("num_drugs", default=0)
Doc.set_extension("num_pers", default=0)
Doc.set_extension("drug_VB_negated", default=False)
Token.set_extension("is_drug", default=False)
Token.set_extension("VB_to_drug", default=False)
Token.set_extension("emoji_text", default=None)
Token.set_extension("pos_rel_drug", default=0)
Token.set_extension("drug_verb_span", default=0)

# New spacy pipeline features
def extract_features(doc):
    drug_pattern = '(dioralyte|solpadine|solpadol|icyhot|zanaflex|tramdol|moldavite|robex|novacaine|tramadol|pedialite|NeoCitran|dydramol|adderal|ciroc|im+odium|profen|excedrine|zanny|gaviscon|cod[ea]mol|panadol|ora[jg]el|ben[ea]dryl|percocet|mucinex|dayquil|nyquil|alieve|sudocrem|hydroco|propionate|fluticasone|liefern|medrol|klonopin|proai|arcoxia|spiriva|anacin|accolate|advai|alprazolam|lorazepam|symbicort|acetaminophen|methadone|oxyc|robitussin|fluoxetine|diazi?e?pam|advil|tylenol|steroid|ibuprofen|motrin|valium|panadol|code?ine|flonase|ativan|proza[ck]|vent[oia]lin|pred(ni|in)sone|xan([enya]x|s)|paracet(am|ma)ol|albuter[oa]l|clona[sz]epam)'
    drug_pos = []

    #Find all the drug tokens
    for pos, token in enumerate(doc):
        if re.match(drug_pattern, token.text, flags=re.I) is not None:
            token._.is_drug = True
            #record index of all drug tokens
            drug_pos.append(pos)

    #Count the PERSON entities
    PERS_count = 0
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            PERS_count += 1

    drug_count = 0
    for pos, token in enumerate(doc):
        #Count the drugs in the doc
        drug_count += int(token._.is_drug)

        #Label verb parent of drug
        if token.pos_ == "VERB":
            for child in token.children:
                if child._.is_drug is True:
                    token._.VB_to_drug = True

        #Is the verb negated?
        if token._.VB_to_drug:
            for child in token.children:
                if child.dep_ == "neg":
                    doc._.drug_VB_negated = True

        #label the token position relative to drug token
        if len(drug_pos)>0:
            dists = [abs(pos-x) for x in drug_pos]
            ind = np.argmin(dists)
            token._.pos_rel_drug = pos-drug_pos[ind]

        #label all drugs within the verb_to_drug subtree
        if token._.is_drug:
            token = token.head
            if token.pos_ != "VERB":
                for child in token.subtree:
                    child._.drug_verb_span = 1

    doc._.num_pers = PERS_count
    doc._.num_drugs = drug_count
    return doc


try:
    nlp.add_pipe(extract_features, name='extract_features', last=True)
except Exception as e:
    nlp.remove_pipe("extract_features")
    nlp.add_pipe(extract_features, name='extract_features', last=True)



# Load Data
# TRAIN_DATA = "./data/train.csv"
# DEV_DATA = "./data/dev.csv"
# TEST_DATA = "./data/test.csv"

# Load Data
TRAIN_DATA = "./data/train_emoji.csv"
DEV_DATA = "./data/dev_emoji.csv"
TEST_DATA = "./data/test_emoji.csv"

# Load data
start_time = time.time()
print("Loading data...")
train = pd.read_csv(TRAIN_DATA)
dev = pd.read_csv(DEV_DATA)
test = pd.read_csv(TEST_DATA)

# Cleaning Text
train["Text"] = train["Text"].apply(lambda x: clean_text(x))
dev["Text"] = dev["Text"].apply(lambda x: clean_text(x))
test["Text"] = test["Text"].apply(lambda x: clean_text(x))


# Spacy Doc
train["Doc"] = train["Text"].apply(lambda x: nlp(x))
dev["Doc"] = dev["Text"].apply(lambda x: nlp(x))
test["Doc"] = test["Text"].apply(lambda x: nlp(x))

print("%s seconds" % (time.time() - start_time))

# Feature Extraction function
def create_emb(doc):
    emb = []
    for token in doc:
        emb.append([int(token._.is_drug),
                   int(token._.VB_to_drug),
                   token._.pos_rel_drug,
                   token._.drug_verb_span])
    return emb

train["Emb"] = train["Doc"].apply(lambda x: create_emb(x))
dev["Emb"] = dev["Doc"].apply(lambda x: create_emb(x))
test["Emb"] = test["Doc"].apply(lambda x: create_emb(x))

# Calculate class weights for model training
print("Calculating Label Weights")
start_time = time.time()
label_weights = {}
temp_weights = []
num_samples = len(train)
for i in range(1,4):
    temp_weights.append(num_samples/train.loc[train["Intake"] == i].count()[0])
total_weight = sum(temp_weights)
for i in range(1,4):
    label_weights[i-1] = temp_weights[i-1]/total_weight

print("%s seconds" % (time.time() - start_time))
label_weights

# Get word indexes
print("Indexing Words")
start_time = time.time()
ind = 1
word_index = {}
lemma_dict = {}
for doc in train["Doc"]:
    for token in doc:
        if token.text not in word_index and token.pos_ is not "PUNCT":
            word_index[token.text] = ind
            ind += 1
            lemma_dict[token.text] = token.lemma_
print("%s seconds" % (time.time() - start_time))


# Load the embedding matrix
start_time = time.time()
import os
GLOVE_DIR = "../embeddings/"


print("Loading glove embedding matrix ...")
embeddings_index = {}
with open(os.path.join(GLOVE_DIR, 'glove.840B.300d.txt'), encoding="utf8") as f:
    for line in f:
        values = line.split(" ")
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

num_words = len(word_index) + 1
embedding_matrix_glove = np.zeros((num_words, 300))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix_glove[i] = embedding_vector

print("%s seconds" % (time.time() - start_time))


print("Loading paragram embedding matrix ...")
embeddings_index = {}
with open(os.path.join(GLOVE_DIR, 'paragram_300_sl999.txt'), encoding="utf8", errors="ignore") as f:
    for line in f:
        values = line.split(" ")
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

num_words = len(word_index) + 1
embedding_matrix_para = np.zeros((num_words, 300))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix_para[i] = embedding_vector

print("%s seconds" % (time.time() - start_time))



print("Loading fasttext embedding matrix ...")
embeddings_index = {}
with open(os.path.join(GLOVE_DIR, 'wiki-news-300d-1M.vec'), encoding="utf8") as f:
    for line in f:
        values = line.split(" ")
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

num_words = len(word_index) + 1
embedding_matrix_fast = np.zeros((num_words, 300))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix_fast[i] = embedding_vector

print("%s seconds" % (time.time() - start_time))



print("Loading twitter embedding matrix ...")
embeddings_index = {}
with open(os.path.join(GLOVE_DIR, 'glove.twitter.27B.200d.txt'), encoding="utf8") as f:
    for line in f:
        values = line.split(" ")
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

num_words = len(word_index) + 1
embedding_matrix_twitter = np.zeros((num_words, 200))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix_twitter[i] = embedding_vector

print("%s seconds" % (time.time() - start_time))

# hyperparameters
max_length = 42

# Padding functions
def pad_seq_doc(df):
    seqs = []
    for doc in df["Doc"]:
        word_seq = []
        for token in doc:
            if token.pos_ != "PUNCT" and token.text in word_index:
                word_seq.append(word_index[token.text])
            elif token.pos_ != "PUNCT":
                word_seq.append(0)
        seqs.append(word_seq)
    return pad_sequences(seqs,maxlen=max_length)

def pad_seq_emb(df):
    seqs = []
    for emb_seq in df["Emb"]:
        dif = max_length - len(emb_seq)
        if dif >0:
            padded_emb = [[0,0,-max_length,0] for x in range(max(0,dif))] + emb_seq
        else:
            padded_emb = emb_seq[:max_length]
        seqs.append(padded_emb)
    return seqs

def extract_doc_features(df):
    features = []
    for doc in df["Doc"]:
        features.append([doc._.num_pers, doc._.num_drugs, doc._.drug_VB_negated])
    return features

# Pad sequences
train_word_sequences = pad_seq_doc(train)
dev_word_sequences = pad_seq_doc(dev)
test_word_sequences = pad_seq_doc(test)

train_emb_sequences = np.array(pad_seq_emb(train))
dev_emb_sequences = np.array(pad_seq_emb(dev))
test_emb_sequences = np.array(pad_seq_emb(test))
train_features = np.array(extract_doc_features(train))
test_features = np.array(extract_doc_features(test))
dev_features = np.array(extract_doc_features(dev))

# Network functions
class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.precision = []
        self.recall = []
        self.f1s = []

    def on_epoch_end(self, epoch, logs={}):
        predict = [x.argmax() for x in np.asarray(self.model.predict(self.validation_data[0]))]
        targ = [x.argmax() for x in self.validation_data[1]]

        self.precision.append(sklm.precision_score(targ, predict, average="micro"))
        self.recall.append(sklm.recall_score(targ, predict, average="micro"))
        self.f1s.append(sklm.f1_score(targ, predict, average="micro"))

        return

metrics = Metrics()

class LayerNormalization(Layer):
    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps
        super(LayerNormalization, self).__init__(**kwargs)
    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer=Ones(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer=Zeros(), trainable=True)
        super(LayerNormalization, self).build(input_shape)
    def call(self, x):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
    def compute_output_shape(self, input_shape):
        return input_shape

class ScaledDotProductAttention():
    def __init__(self, d_model, attn_dropout=0.1):
        self.temper = np.sqrt(d_model)
        self.dropout = Dropout(attn_dropout)
    def __call__(self, q, k, v, mask):
        attn = Lambda(lambda x:K.batch_dot(x[0],x[1],axes=[2,2])/self.temper)([q, k])
        if mask is not None:
            mmask = Lambda(lambda x:(-1e+10)*(1-x))(mask)
            attn = Add()([attn, mmask])
        attn = Activation('softmax')(attn)
        attn = self.dropout(attn)
        output = Lambda(lambda x:K.batch_dot(x[0], x[1]))([attn, v])
        return output, attn

class MultiHeadAttention():
    # mode 0 - big martixes, faster; mode 1 - more clear implementation
    def __init__(self, n_head, d_model, d_k, d_v, dropout, mode=0, use_norm=True):
        self.mode = mode
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = dropout
        if mode == 0:
            self.qs_layer = Dense(n_head*d_k, use_bias=False)
            self.ks_layer = Dense(n_head*d_k, use_bias=False)
            self.vs_layer = Dense(n_head*d_v, use_bias=False)
        elif mode == 1:
            self.qs_layers = []
            self.ks_layers = []
            self.vs_layers = []
            for _ in range(n_head):
                self.qs_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
                self.ks_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
                self.vs_layers.append(TimeDistributed(Dense(d_v, use_bias=False)))
        self.attention = ScaledDotProductAttention(d_model)
        self.layer_norm = LayerNormalization() if use_norm else None
        self.w_o = TimeDistributed(Dense(d_model))

    def __call__(self, q, k, v, mask=None):
        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head

        if self.mode == 0:
            qs = self.qs_layer(q)  # [batch_size, len_q, n_head*d_k]
            ks = self.ks_layer(k)
            vs = self.vs_layer(v)

            def reshape1(x):
                s = tf.shape(x)   # [batch_size, len_q, n_head * d_k]
                x = tf.reshape(x, [s[0], s[1], n_head, d_k])
                x = tf.transpose(x, [2, 0, 1, 3])
                x = tf.reshape(x, [-1, s[1], d_k])  # [n_head * batch_size, len_q, d_k]
                return x
            qs = Lambda(reshape1)(qs)
            ks = Lambda(reshape1)(ks)
            vs = Lambda(reshape1)(vs)

            if mask is not None:
                mask = Lambda(lambda x:K.repeat_elements(x, n_head, 0))(mask)
            head, attn = self.attention(qs, ks, vs, mask=mask)

            def reshape2(x):
                s = tf.shape(x)   # [n_head * batch_size, len_v, d_v]
                x = tf.reshape(x, [n_head, -1, s[1], s[2]])
                x = tf.transpose(x, [1, 2, 0, 3])
                x = tf.reshape(x, [-1, s[1], n_head*d_v])  # [batch_size, len_v, n_head * d_v]
                return x
            head = Lambda(reshape2)(head)
        elif self.mode == 1:
            heads = []; attns = []
            for i in range(n_head):
                qs = self.qs_layers[i](q)
                ks = self.ks_layers[i](k)
                vs = self.vs_layers[i](v)
                head, attn = self.attention(qs, ks, vs, mask)
                heads.append(head); attns.append(attn)
            head = Concatenate()(heads) if n_head > 1 else heads[0]
            attn = Concatenate()(attns) if n_head > 1 else attns[0]

        outputs = self.w_o(head)
        outputs = Dropout(self.dropout)(outputs)
        if not self.layer_norm: return outputs, attn
        # outputs = Add()([outputs, q]) # sl: fix
        return self.layer_norm(outputs), attn


def build_simple_LSTM(embedding_matrix, nb_words, embedding_size=300, extra_embs=[0,0,0,0], doc_features=[0,0,0]):
    inp_words = Input(shape=(max_length,))
    inputs = [inp_words]
    if sum(extra_embs)>0:
        inp_embs = Input(shape=(max_length,sum(extra_embs),))
        inputs.append(inp_embs)
    if sum(doc_features) > 0:
        inp_docs = Input(shape=(sum(doc_Features),))
        inputs.append(inp_docs)

    x = Embedding(nb_words, embedding_size, trainable=False, embeddings_initializer=Constant(embedding_matrix),)(inp_words)

    if sum(extra_embs) > 0:
        x = Concatenate(axis=2)([x,inp_embs])

    # x1 = Bidirectional(LSTM(32, return_sequences=False))(x)
    x1 = LSTM(32, return_sequences=False)(x)
    if sum(doc_features) > 0:
        x1 = Concatenate()([x1,inp_docs])
    d = Dropout(0.3)(x1)

    predictions = Dense(3, activation='softmax')(d)

    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_yookim_CNN(embeddings_matrix, nb_words, embedding_size=300, extra_embs=[0,0,0,0], doc_features=[0,0,0]):
    K.clear_session()
    inp_words = Input(shape=(max_length,), dtype='int32')
    inputs = [inp_words]
    if sum(extra_embs)>0:
        inp_embs = Input(shape=(max_length,sum(extra_embs),))
        inputs.append(inp_embs)
    if sum(doc_features) > 0:
        inp_docs = Input(shape=(sum(doc_features),))
        inputs.append(inp_docs)

    x = Embedding(nb_words, embedding_size, trainable=False, embeddings_initializer=Constant(embedding_matrix),)(inp_words)

    if sum(extra_embs) > 0:
        x = Concatenate(axis=2)([x,inp_embs])

    convs = []
    filter_sizes = [3,4,5]
    for fsz in filter_sizes:
        l_conv = Conv1D(filters=128,kernel_size=fsz,activation='relu')(x)
        l_pool = MaxPooling1D(5)(l_conv)
        convs.append(l_pool)
    l_merge = concatenate(convs, axis=1)
    l_cov1= Conv1D(128, 5, activation='relu', data_format = 'channels_first')(l_merge)
    l_pool1 = MaxPooling1D(5)(l_cov1)
    l_cov2 = Conv1D(128, 5, activation='relu', data_format = 'channels_first')(l_pool1)
    l_pool2 = MaxPooling1D(30)(l_cov2)
    l_flat = Flatten()(l_pool2)
    if sum(doc_features) > 0:
        l_flat = Concatenate()([l_flat,inp_docs])
    l_dense = Dense(128, activation='relu')(l_flat)
    predictions = Dense(3, activation='softmax')(l_dense)

    model = Model(inputs=inputs, outputs=predictions)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=["acc"])
    return model

def build_attention_LSTM(embeddings_matrix, nb_words, embedding_size=300, extra_embs=[0,0,0,0], doc_features=[0,0,0], binary=False, no_embeddings=False):
    inputs = []
    if not no_embeddings:
        inp_words = Input(shape=(max_length,))
        x = Embedding(nb_words, embedding_size, trainable=False, embeddings_initializer=Constant(embedding_matrix),)(inp_words)
        inputs.append(inp_words)
    if sum(extra_embs)>0:
        inp_embs = Input(shape=(max_length,sum(extra_embs),))
        if not no_embeddings:
            x = Concatenate(axis=2)([x,inp_embs])
        else: x = inp_embs
        inputs.append(inp_embs)
    if sum(doc_features) > 0:
        inp_docs = Input(shape=(sum(doc_features),))
        inputs.append(inp_docs)

    # LSTM before attention layers
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)

    x, slf_attn = MultiHeadAttention(n_head=3, d_model=300, d_k=64, d_v=64, dropout=0.3)(x, x, x)

    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool],axis=1)
    if sum(doc_features) > 0:
        x1 = Concatenate()([conc,inp_docs])
    conc = Dense(64, activation="relu")(conc)
    if binary:
        preds = Dense(2, activation="softmax")(conc)
    else:
        preds = Dense(3, activation="softmax")(conc)

    model = Model(inputs=inputs, outputs=preds)
    model.compile(
        loss = "categorical_crossentropy",
        optimizer = "adam")

    return model


def build_model(embedding_matrix, nb_words, embedding_size=300, extra_embs=[0,0,0,0], model_type="simple", doc_features=[0,0,0], binary=False, no_embeddings=False):
    if model_type == "simple":
        return build_simple_LSTM(embedding_matrix, nb_words, embedding_size, extra_embs, doc_features)
    elif model_type == "cnn":
        return build_yookim_CNN(embedding_matrix, nb_words, embedding_size, extra_embs, doc_features)
    elif model_type == "lstm":
        return build_attention_LSTM(embedding_matrix, nb_words, embedding_size, extra_embs, doc_features, binary, no_embeddings)

def make_binary(value):
    if value == 3:
        return 2
    else:
        return 1

embedding_matrix = np.concatenate((embedding_matrix_twitter, embedding_matrix_para), axis=1)
embedding_matrix = np.concatenate((embedding_matrix, embedding_matrix_fast), axis=1)
embedding_matrix = np.concatenate((embedding_matrix, embedding_matrix_glove), axis=1)
# embedding_matrix = embedding_matrix_fast

no_embeddings = False
binary = False

if binary:
    # To categorical
    train["Binary"] = train["Intake"].apply(make_binary)
    test["Binary"] = test["Intake"].apply(make_binary)
    dev["Binary"] = dev["Intake"].apply(make_binary)

    y_train = to_categorical(train['Binary'].values)[:,[1,2]]
    y_dev = to_categorical(dev['Binary'].values)[:,[1,2]]
    y_test = to_categorical(test['Binary'].values)[:,[1,2]]
    outs = 2
    outs_y = "Binary"
else:
    outs = 3
    outs_y = "Intake"
    # To categorical
    y_train = to_categorical(train['Intake'].values)[:,[1,2,3]]
    y_dev = to_categorical(dev['Intake'].values)[:,[1,2,3]]
    y_test = to_categorical(test['Intake'].values)[:,[1,2,3]]
# is_drug, VB_to_drug, pos_rel_drug, drug_verb_span
# num_pers,num_drugs,vb_negated

#semantic [1,0,0,0],[1,1,0] + emoji
#syntactic [0,1,0,0],[0,0,1]
# feat_dict = {"none": ([0,0,0,0],[0,0,0]),
# feat_dict= {"semantic": ([1,0,0,0],[1,1,0]),
            # "syntactic": ([0,1,0,0],[0,0,1]),
            # "pseudo1": ([0,0,1,0],[0,0,0]),
            # "pseudo2": ([0,0,0,1],[0,0,0]),
feat_dict={"pseudo3": ([0,0,1,1],[0,0,0]),
            # "semantic+syntactic": ([1,1,0,0],[1,1,1]),
            # "semantic+pseudo1": ([1,0,1,0],[1,1,0]),
            # "semantic+pseudo2": ([1,0,0,1],[1,1,0]),
            # "semantic+pseudo3": ([1,0,1,1],[1,1,0]),
            # "syntactic+pseudo1": ([0,1,1,0],[0,0,1]),
            # "syntactic+pseudo2": ([0,1,0,1],[0,0,1]),
            # "syntactic+pseudo3": ([0,1,1,1],[0,0,1]),
            # "semantic+syntactic+pseudo1": ([1,1,1,0],[1,1,1]),
            # "semantic+syntactic+pseudo2": ([1,1,0,1],[1,1,1]),
            # "semantic+syntactic+pseudo3": ([1,1,1,1],[1,1,1]),
}

for feat_type in feat_dict.keys():

    embeddings = "all"
    embs = feat_dict[feat_type][0]
    doc_features = feat_dict[feat_type][1]
    embs_ind = [ind for ind,x in enumerate(embs) if x>0]
    feats_ind = [ind for ind,x in enumerate(doc_features) if x>0]
    batch_size = 32
    num_epoch = 10
    runs = 10
    model_type = "lstm"
    fp = "outputs/{}_{}_{}_{}".format(model_type,feat_type,embeddings,binary)
    if no_embeddings:
        fp += "_no-embeddings"

    train_input = []
    test_input = []
    dev_input = []

    if not no_embeddings:
        train_input = [train_word_sequences]
        test_input = [test_word_sequences]
        dev_input = [dev_word_sequences]

    if len(embs_ind) > 0:
        train_input.append(train_emb_sequences[:,:,embs_ind])
        test_input.append(test_emb_sequences[:,:,embs_ind])
        dev_input.append(dev_emb_sequences[:,:,embs_ind])
    if len(feats_ind) >0:
        train_input.append(train_features[:,feats_ind])
        test_input.append(test_features[:,feats_ind])
        dev_input.append(dev_features[:,feats_ind])


    with open(fp,"w") as f:
        f.write("{}\n{} runs\n{} epochs each\nbatch size{}\nfeatures {}".format(model_type,runs,num_epoch,batch_size,embs))
        print("Start training ...")
        pred_prob = np.zeros((len(test_word_sequences),outs), dtype=np.float32)
        start_time = time.time()
        for run in range(runs):
            print("\n\n\n\n RUN {}".format(run))
            K.clear_session()
            model = build_model(embedding_matrix, num_words, embedding_matrix.shape[1],extra_embs=embs, model_type=model_type, doc_features=doc_features, binary=binary, no_embeddings=no_embeddings)
            # checkpoint = ModelCheckpoint(fp+".weights", monitor='val_loss', verbose=0, save_best_only=True, mode='min')
            es = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto', restore_best_weights=True)
            callbacks_list = [es]
            model.fit(train_input,
                      y_train,
                      validation_data=[dev_input,y_dev],
                      batch_size=batch_size,
                      epochs=num_epoch,
                      class_weight=label_weights,
                      callbacks=callbacks_list,
                      verbose=3)

            pred_prob += 1/runs * np.squeeze(model.predict(test_input, batch_size=batch_size, verbose=3))
            del model


        print("--- %s seconds ---" % (time.time() - start_time))
        preds = [x.argmax() + 1 for x in pred_prob]
        f.write("\n\n\nTest Results\n")
        f.write(classification_report(test[outs_y].values, preds, digits=4))
        tp11, fp12, fp13, fp21, tp22, fp23, fp31, fp32, tp33 = sklm.confusion_matrix(test[outs_y].values, preds).ravel()
        P12 = (tp11 + tp22) / (tp11 + fp12 + fp13 + tp22 + fp21 + fp23)
        R12 = (tp11 + tp22) / (tp11 + fp21 +fp31 + tp22 + fp12 + fp32)
        F12 = (2 * P12 * R12) / (P12 + R12)
        f.write("Micro Averaged:")
        f.write("\nP:{:.4f}\tR:{:.4f}\tF:{:.4f}".format(P12,R12,F12))
