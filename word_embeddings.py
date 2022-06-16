from gensim.models import Word2Vec, word2vec
import matplotlib.pyplot as plt
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
from nltk.corpus import stopwords
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import string
import re
import glob


def review_to_wordlist(review):
    review_text = re.sub("[^a-zA-Z]", " ", review)
    words = review_text.lower().split()

    stops = set(stopwords.words("english"))
    words = [w for w in words if not w in stops]
    return words


def review_to_sentences(review, tokenizer):
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
    if len(glob.glob("cleanedup.csv")) == 0:
        df = pd.read_csv(path[0])
        df = df[df.Score != 3]
        df['Text'] = df['Text'].apply(process)
        df.to_csv("cleanedup.csv")
    else:
        df = pd.read_csv("cleanedup.csv")
    return df


def make_feature_vec(words, model, num_features):
    """
    Average the word vectors for a set of words
    """
    feature_vec = np.zeros(
        (num_features,), dtype="float32")  # pre-initialize (for speed)
    nwords = 0
    index2word_set = set(model.wv.index_to_key)  # words known to the model

    for word in words:
        if word in index2word_set:
            nwords = nwords + 1
            feature_vec = np.add(feature_vec, model.wv[word])
    if nwords > 0:
        feature_vec = np.divide(feature_vec, nwords)
    return feature_vec


def get_avg_feature_vecs(reviews, model, num_features):
    """
    Calculate average feature vectors for all reviews
    """
    counter = 0
    # pre-initialize (for speed)
    review_feature_vecs = np.zeros(
        (len(reviews), num_features), dtype='float32')

    for review in reviews:
        review_feature_vecs[counter] = make_feature_vec(
            review, model, num_features)
        counter = counter + 1
    return review_feature_vecs


if __name__ == "__main__":
    datapath = "amazon.csv"
    original = pd.read_csv(datapath)
    original.hist(column='Score')
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    df = readCSV(datapath)
    df.hist(column="Score")
    plt.show()
    df['Class'] = 1 * (df['Score'] > 3)
    train_size = int(len(df) * 0.8)
    train_reviews = df.iloc[:train_size, :]
    test_reviews = df.iloc[train_size:, :]
    print(f'Training set contains {len(train_reviews):d} reviews.')
    print(f'Test set contains {len(test_reviews):d} reviews')
    n_pos_train = sum(train_reviews['Class'] == 1)
    print(
        f'Training set contains {(n_pos_train/len(train_reviews)):.2%} positive reviews')
    n_pos_test = sum(test_reviews['Class'] == 1)
    print('Test set contains {:.2%} positive reviews'.format(
        n_pos_test/len(test_reviews)))
    train_sentences = []
    for review in train_reviews['Text']:
        review = " ".join(review)
        train_sentences += review_to_sentences(review, tokenizer)

    model_name = "MyWord2Vec.model"
    # Set values for various word2vec parameters
    num_features = 400    # Word vector dimensionality
    min_word_count = 150   # Minimum word count
    num_workers = 4      # Number of threads to run in parallel
    context = 10          # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words
    if not os.path.exists(model_name):
        # Initialize and train the model (this will take some time)
        model = word2vec.Word2Vec(train_sentences, workers=num_workers,
                                  vector_size=num_features, min_count=min_word_count,
                                  window=context, sample=downsampling)

        model.save(model_name)
    else:
        model = Word2Vec.load(model_name)

    clean_train_reviews = []
    for review in train_reviews['Text']:
        clean_train_reviews.append(
            review_to_wordlist(review))
    trainDataVecs = get_avg_feature_vecs(
        clean_train_reviews, model, num_features)

    clean_test_reviews = []
    for review in test_reviews['Text']:
        clean_test_reviews.append(
            review_to_wordlist(review))
    testDataVecs = get_avg_feature_vecs(
        clean_test_reviews, model, num_features)
    estimators = [10, 50, 100, 200, 500]
    accuracy = []
    for n_estimators in estimators:
        forest = RandomForestClassifier(
            n_estimators=n_estimators, class_weight='balanced_subsample')
        print("Fitting a random forest to labeled training data...")
        forest = forest.fit(trainDataVecs, train_reviews['Class'])

        # remove instances in test set that could not be represented as feature vectors
        nan_indices = list({x for x, y in np.argwhere(np.isnan(testDataVecs))})
        if len(nan_indices) > 0:
            print(f'Removing {len(nan_indices)} instances from test set.')
            testDataVecs = np.delete(testDataVecs, nan_indices, axis=0)
            test_reviews.drop(
                test_reviews.iloc[nan_indices, :].index, axis=0, inplace=True)
            assert testDataVecs.shape[0] == len(test_reviews)

        print(f"Estimators used: {n_estimators}")
        print("Predicting labels for test data..")
        result = forest.predict(testDataVecs)

        print(classification_report(test_reviews['Class'], result))
        accuracy.append(classification_report(test_reviews['Class'], result))
        probs = forest.predict_proba(testDataVecs)[:, 1]

        fpr, tpr, _ = roc_curve(test_reviews['Class'], probs)
        auc = roc_auc_score(test_reviews['Class'], probs)

        plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr, label=f'AUC {auc:.3f}')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        plt.show()
