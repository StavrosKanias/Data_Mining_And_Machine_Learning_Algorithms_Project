from gensim.models import Word2Vec, word2vec
import logging
import matplotlib.pyplot as plt
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
from nltk.corpus import stopwords
import numpy as np
import os
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
import string
import re
from tqdm import tqdm
import glob

MAX_LEN = 150
MAX_WORDS = 400


def review_to_wordlist(review):
    review_text = re.sub("[^a-zA-Z]", " ", review)
    words = review_text.lower().split()

    stops = set(stopwords.words("english"))
    words = [w for w in words if not w in stops]
    return words


def review_to_sentences(review, tokenizer, remove_stopwords=False):
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(review_to_wordlist(raw_sentence))
    return sentences


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
        df = df[df.Score != 3]
        df['Text'] = df['Text'].apply(process)
        df.to_csv("cleanedup.csv")
    else:
        df = pd.read_csv("cleanedup.csv")
    return df


# def train_model(df):
#     X = df.Text
#     Y = df.Score

#     X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.2)

#     # Tokenizer
#     tok = Tokenizer(num_words=MAX_WORDS)
#     tok.fit_on_texts(X_Train)
#     sequences = tok.texts_to_sequences(X_Train)
#     sequences_matrix = pad_sequences(sequences, maxlen=MAX_LEN)

#     tok = Tokenizer(num_words=MAX_WORDS)
#     tok.fit_on_texts(X_Test)
#     sequences = tok.texts_to_sequences(X_Test)
#     sequences_matrix_test = pad_sequences(sequences, maxlen=MAX_LEN)

#     return X_Train, X_Test, Y_Train, Y_Test, sequences_matrix, sequences_matrix_test


# def random_forest(dataset):
#     x_train, x_test, y_train, y_test, seq1, seq2 = train_model(dataset)
#     clf = RandomForestClassifier(
#         random_state=0, n_estimators=100, max_features=1.0, bootstrap=False)
#     clf.fit(seq1, y_train)
#     predictions = clf.predict(seq2)
#     return y_test, predictions


# def evaluate(y_test, predictions):
#     recall = recall_score(y_test, predictions, average='macro')
#     precision = precision_score(y_test, predictions, average='macro')
#     f1 = f1_score(y_test, predictions, average='macro')
#     true_pos = 0
#     true_neg = 0
#     false_pos = 0
#     false_neg = 0
#     for i in range(y_test.size):
#         if predictions[i] == y_test[i] and y_test[i] == 1:
#             true_pos += 1
#         if predictions[i] == y_test[i] and y_test[i] == 0:
#             true_neg += 1
#         if predictions[i] != y_test[i] and predictions[i] == 1:
#             false_pos += 1
#         if predictions[i] != y_test[i] and predictions[i] == 0:
#             false_neg += 1
#     print({"true pos": true_pos, "false pos": false_pos})
#     print({"false neg": false_neg, "true neg": true_neg})
#     return {"precision": precision, "recall": recall, "f1": f1}


if __name__ == "__main__":
    datapath = "amazon.csv"
    df = readCSV(datapath)
    df.hist(column="Score")
    plt.show()
