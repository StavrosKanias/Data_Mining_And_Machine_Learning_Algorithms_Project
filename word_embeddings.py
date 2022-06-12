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

from keras.optimizers import RMSprop, Adam

from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split


def readCSV(datapath):
    path = glob.glob(datapath)
    print(path)
    pass


if __name__ == "__main__":
    datapath = "amazon.csv"
    readCSV(datapath)
