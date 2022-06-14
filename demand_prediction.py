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


def create_dataset():
    df = pd.read_csv("unified.csv")
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.set_index('Datetime')
    df = df.astype(np.float64).fillna(method='bfill')

    # print(df)

    # For simplication,
    # I will resample so that each row
    # represents a whole hour
    df_uci_daily = df.resample('d').sum()
    df_uci_daily['day'] = df_uci_daily.index.day
    df_uci_daily.index = df_uci_daily.index.date
    df_uci_daily.drop(['day'], axis=1, inplace=True)
    return df_uci_daily


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    dff = pd.DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(dff.shift(-i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(dff.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1)) for j in range(n_vars)]
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        if dropnan:
            agg.dropna(inplace=True)
        return agg


if __name__ == "__main__":
    df = create_dataset()
