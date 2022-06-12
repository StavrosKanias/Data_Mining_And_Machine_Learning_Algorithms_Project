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

    max_words = 700
    max_len = 150

    tok = Tokenizer(num_words=max_words)
    tok.fit_on_texts(X_Train)
    sequences = tok.texts_to_sequences(X_Train)
    sequences_matrix = pad_sequences(sequences, maxlen=max_len)

    return sequences_matrix


if __name__ == "__main__":
    datapath = "amazon.csv"
    df = readCSV(datapath)
