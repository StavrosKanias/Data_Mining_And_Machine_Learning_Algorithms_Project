from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sillhoute_scores = []
n_cluster_list = np.arange(2, 31).astype(int)


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


df_uci_pivot = create_pivot()
X = df_uci_pivot.copy()

# Very important to scale!
sc = MinMaxScaler()
X = sc.fit_transform(X)

for n_cluster in n_cluster_list:

    kmeans = KMeans(n_clusters=n_cluster)
    cluster_found = kmeans.fit_predict(X)
    sillhoute_scores.append(silhouette_score(X, kmeans.labels_))

# plt.plot(silhouette_score)
# plt.show()
kmeans = KMeans(n_clusters=3)
cluster_found = kmeans.fit_predict(X)
cluster_found_sr = pd.Series(cluster_found, name='cluster')
df_uci_pivot = df_uci_pivot.set_index(cluster_found_sr, append=True)

fig, ax = plt.subplots(1, 1, figsize=(18, 10))
color_list = ['blue', 'red', 'green']
cluster_values = sorted(
    df_uci_pivot.index.get_level_values('cluster').unique())

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
plt.show()
