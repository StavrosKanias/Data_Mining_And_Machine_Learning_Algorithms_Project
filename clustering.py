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
import matplotlib.colors


def tuneParams(X):

    # Defining the list of hyperparameters to try
    eps_list = np.arange(start=0.1, stop=10.0, step=0.1)
    min_sample_list = np.arange(start=2, stop=20, step=1)

    # Creating empty data frame to store the silhouette scores for each trials
    silhouette_scores_data = pd.DataFrame()

    for eps_trial in eps_list:
        for min_sample_trial in min_sample_list:

            # Generating DBSAN clusters
            db = DBSCAN(eps=eps_trial, min_samples=min_sample_trial)

            if(len(np.unique(db.fit_predict(X))) > 1):
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
    return tuned_eps, tuned_min_samples


def create_pivot():
    df = pd.read_csv("summedUp.csv")
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')

    print(df)

    df = df.astype(np.float64).fillna(method='bfill')

    # For simplication,
    # I will resample so that each row
    # represents a whole hour
    df_uci_hourly = df.resample('H').sum()
    df_uci_hourly['hour'] = df_uci_hourly.index.hour
    df_uci_hourly.index = df_uci_hourly.index.date

    print(df_uci_hourly)

    df_uci_pivot = df_uci_hourly.pivot(columns='hour')
    df_uci_pivot = df_uci_pivot.dropna()
    return df_uci_pivot


def cluster_KMeans():
    sillhoute_scores = []
    n_cluster_list = np.arange(2, 31).astype(int)
    df_uci_pivot = create_pivot()
    X = df_uci_pivot.copy()

    # Very important to scale!
    sc = MinMaxScaler()
    X = sc.fit_transform(X)

    for n_cluster in n_cluster_list:

        kmeans = KMeans(n_clusters=n_cluster)
        cluster_found = kmeans.fit_predict(X)
        sillhoute_scores.append(silhouette_score(X, kmeans.labels_))

    plt.figure(0)
    plt.plot(sillhoute_scores)

    kmeans = KMeans(n_clusters=3)
    cluster_found = kmeans.fit_predict(X)
    cluster_found_sr = pd.Series(cluster_found, name='cluster')
    df_uci_pivot = df_uci_pivot.set_index(cluster_found_sr, append=True)
    cluster_values = sorted(
        df_uci_pivot.index.get_level_values('cluster').unique())

    return X, cluster_values, df_uci_pivot


def plot_sillhouete_clusers(X, cluster_values, color_list, df_uci_pivot):

    fig, ax = plt.subplots(1, 1, figsize=(18, 10))

    for cluster, color in zip(cluster_values, color_list):
        df_uci_pivot.xs(cluster, level=1).T.plot(
            ax=ax, legend=False, alpha=0.01, color=color, label=f'Cluster {cluster}'
        )
        df_uci_pivot.xs(cluster, level=1).median().plot(
            ax=ax, color=color, alpha=0.9, ls='--'
        )
    ax.set_xticks(np.arange(1, 25))
    ax.set_ylabel('kilowatts')
    ax.set_xlabel('hour')


def plot_scatter_clusters(X, cluster_values, color_list, df_uci_pivot):

    tsne = TSNE()
    results_tsne = tsne.fit_transform(X)

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        cluster_values, color_list)
    plt.scatter(results_tsne[:, 0], results_tsne[:, 1],
                c=df_uci_pivot.index.get_level_values('cluster'),
                cmap=cmap,
                alpha=0.6,
                )
    return results_tsne


def our_dbscan():
    X = create_pivot()
    tsne = TSNE(random_state=1)
    X = tsne.fit_transform(X)
    eps, min_samples = tuneParams(X)
    nbrs = NearestNeighbors(n_neighbors=min_samples).fit(X)

    neist_dist, neist_ind = nbrs.kneighbors(X)

    sort_neight_dist = np.sort(neist_dist, axis=0)

    k_dist = sort_neight_dist[:, 10]
    plt.figure(0)
    plt.plot(k_dist)
    plt.axhline(y=eps, linewidth=1, linestyle='dashed', color='k')

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)
    print("Silhouette Coefficient: %0.3f" %
          metrics.silhouette_score(X, labels))
    unique_labels = set(labels)
    plt.figure(1)
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
    plt.show()

    return X


color_list = ['blue', 'red', 'green']
# X, cluster_values, df_uci_pivot = cluster_KMeans()
x = our_dbscan()
# plot_sillhouete_clusers(X, cluster_values, color_list, df_uci_pivot)
# plt.figure(2)
# plot_scatter_clusters(X, cluster_values, color_list, df_uci_pivot)
# plt.show()
