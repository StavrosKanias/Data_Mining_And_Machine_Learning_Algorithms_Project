from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from collections import Counter
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors


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


def our_dbscan(X):

    tsne = TSNE()
    X = tsne.fit_transform(X)

    nbrs = NearestNeighbors(n_neighbors=5).fit(X)

    neist_dist, neist_ind = nbrs.kneighbors(X)

    sort_neight_dist = np.sort(neist_dist, axis=0)

    k_dist = sort_neight_dist[:, 4]
    plt.figure(3)
    plt.plot(k_dist)
    plt.axhline(y=2.5, linewidth=1, linestyle='dashed', color='k')
    plt.show()

    clusters = DBSCAN(eps=2.5, min_samples=5).fit(X)
    print('Unique clusters:')
    print(set(clusters.labels_))
    print('Cluster sizes:')
    print(Counter(clusters.labels_))

    p = sns.scatterplot(data=X, palette='deep')
    sns.move_legend(p,   'upper right', bbox_to_anchor=(
        1.17, 1.2), title='Clusters')
    plt.show()
    return X


color_list = ['blue', 'red', 'green']
X, cluster_values, df_uci_pivot = cluster_KMeans()
x = our_dbscan(df_uci_pivot)
# plot_sillhouete_clusers(X, cluster_values, color_list, df_uci_pivot)
# plt.figure(2)
# plot_scatter_clusters(X, cluster_values, color_list, df_uci_pivot)
# plt.show()
