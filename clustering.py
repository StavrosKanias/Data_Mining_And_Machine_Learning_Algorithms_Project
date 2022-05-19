from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
from datetime import datetime
plt.style.use('seaborn')

data_path = 'demand/*'
df = pd.DataFrame()

csvs = glob.glob(data_path)
osSlash = csvs[0][6]

for i, csv in enumerate(csvs):
    year, month, day = csv.split(osSlash)[1][:4], csv.split(osSlash)[
        1][4:6], csv.split(osSlash)[1][6:8]
    # Check if day exists
    try:
        date = datetime.strptime(
            month + '/' + day + '/' + year, '%m/%d/%Y')
    except ValueError:
        continue
    df_uci = pd.read_csv(csv)
    # print(df_uci)
    if i == 0:
        df['Time'] = df_uci['Time']
    df[f'Current demand {i}'] = df_uci["Current demand"]

print(df)
