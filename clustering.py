from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from collections import Counter
import seaborn as sns
import numpy as np
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt


def tuneParams(X):

    # Defining the list of hyperparameters to try
    eps_list = np.arange(start=550, stop=650, step=1)
    min_sample_list = np.arange(start=2, stop=20, step=1)

    # Creating empty data frame to store the silhouette scores for each trials
    silhouette_scores_data = pd.DataFrame()

    for eps_trial in eps_list:
        for min_sample_trial in min_sample_list:

            # Generating DBSAN clusters
            db = DBSCAN(eps=eps_trial, min_samples=min_sample_trial)

            if(len(np.unique(db.fit_predict(X)))
               > 1):
                sil_score = silhouette_score(X, db.fit_predict(X))
            else:
                continue
            trial_parameters = (eps_trial.round(2), min_sample_trial)

            silhouette_scores_data = pd.concat([silhouette_scores_data, pd.DataFrame(
                data=[[sil_score, trial_parameters]], columns=["score", "parameters"])])

    # Finding out the best hyperparameters with highest Score
    tunedParams = silhouette_scores_data.sort_values(
        by='score', ascending=False)['parameters'].iloc[0]
    tuned_eps = tunedParams[0]
    tuned_min_samples = tunedParams[1]
    bestScore = silhouette_scores_data.sort_values(
        by='score', ascending=False)['score'].iloc[0]

    return tuned_eps, tuned_min_samples, bestScore


def create_dataset():
    df = pd.read_csv("unified.csv")
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.set_index('Datetime')
    df = df.astype(np.float64).fillna(method='bfill')

    # print(df)

    # For simplication,
    # I will resample so that each row
    # represents a whole hour
    df_uci_daily = df.resample('d').mean()
    df_uci_daily['day'] = df_uci_daily.index.day
    df_uci_daily.index = df_uci_daily.index.date
    return df_uci_daily


def our_dbscan(X, title):
    newX = X['Demand'].values
    newY = X['Supply'].values
    print(newX)
    print(newY)
    temp = []
    for x, y in zip(newX, newY):
        temp.append([x, y])
    print(X)
    X = np.array(temp)
    print(X)
    eps, min_samples, silhouette_score = tuneParams(X)
    nbrs = NearestNeighbors(n_neighbors=min_samples).fit(X)

    neist_dist, neist_ind = nbrs.kneighbors(X)

    sort_neight_dist = np.sort(neist_dist, axis=0)

    k_dist = sort_neight_dist[:, 3]
    plt.figure()
    plt.plot(k_dist)
    plt.axhline(y=eps, linewidth=1, linestyle='dashed', color='k')
    plt.savefig('Optimal eps for ' + title + '.png')

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)
    print("Silhouette Coefficient: %0.3f" % silhouette_score)
    unique_labels = set(labels)
    plt.figure()
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = labels == k

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=14,
        )

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=6,
            label=k
        )

    plt.title("Estimated number of clusters: %d" % n_clusters_)
    plt.legend()
    plt.savefig('DBscan clusters for ' + title + '.png')
    # plt.show()

    return X, db


color_list = ['blue', 'red', 'green']
df = create_dataset()
x, db = our_dbscan(df, 'test')
outIndex = []
for i, label in enumerate(db.labels_):
    if label == -1:
        outIndex.append(i)
for i in outIndex:
    print(i)
