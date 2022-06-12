import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import glob

from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.layers.embeddings import Embedding

from tensorflow.keras.optimizers import RMSprop, Adam

from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split

MAX_LEN = 150
MAX_WORDS = 700


def process(text):
    lemmatizer = WordNetLemmatizer()

    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc).lower()

    clean = [lemmatizer.lemmatize(word) for word in nopunc.split(
    ) if word.lower() not in stopwords.words('english')]

    return clean


def readCSV(datapath):
    path = glob.glob(datapath)
    # print(path)
    if len(glob.glob("cleanedup.csv")) == 0:
        df = pd.read_csv(path[0])
        df['Text'] = df['Text'].apply(process)
        df.to_csv("cleanedup.csv")
    else:
        df = pd.read_csv("cleanedup.csv")
    return df


def train_model(df):
    X = df.Text
    Y = df.Score

    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.2)

    # Tokenizer

    tok = Tokenizer(num_words=MAX_WORDS)
    tok.fit_on_texts(X_Train)
    sequences = tok.texts_to_sequences(X_Train)
    sequences_matrix = pad_sequences(sequences, maxlen=MAX_LEN)

    return sequences_matrix, Y_Train


def RNN():

    inputs = Input(name='Input_LAYER', shape=[MAX_LEN])
    layer = Embedding(MAX_WORDS, MAX_LEN, input_length=(
        MAX_LEN, MAX_WORDS), trainable=False, name='embedding')(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256, name='dense1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1, name='output')(layer)
    layer = Dense(1, activation='sigmoid', name='predictions')(layer)
    model = Model(inputs=inputs, outputs=layer)
    return model


if __name__ == "__main__":
    datapath = "amazon.csv"
    df = readCSV(datapath)
