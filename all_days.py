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
    # print(csv)
    # print(df_uci["Current demand"])

# df_uci['datetime'] = pd.to_datetime(df_uci['Date'] + ' ' + df_uci['Time'])
# df_uci = df_uci.drop(['Date', 'Time'], axis=1)
# df_uci = df_uci.set_index('datetime')

# df = df.replace('?', np.nan)
# df = df.astype(np.float).fillna(method='bfill')

# For simplication,
# I will resample so that each row
# represents a whole hour
# df_uci_hourly = df.resample('H').sum()
# df_uci_hourly['hour'] = df_uci_hourly.index.hour
# df_uci_hourly.index = df_uci_hourly.index.date

# df_uci_pivot = df_uci_hourly.pivot(columns='hour')
# df_uci_pivot = df_uci_pivot.dropna()
# print(df)
df.plot(figsize=(13, 8), legend=False, color='blue', alpha=0.02)
plt.show()
# print(len(csvs))
