import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
from datetime import datetime


def hourly(df, column):
    hourlyMean = []
    temp = 0
    for i, demand in enumerate(df[column]):
        if i % 12 == 0 and i != 0:
            hourlyMean.append(temp)
            temp = 0
        temp += demand
    return hourlyMean


plt.style.use('seaborn')

data_path = 'demand/*'
df = pd.DataFrame()

csvs = glob.glob(data_path)
csvs.sort()
osSlash = csvs[0][6]
if len(glob.glob("summedUp.csv")) == 0:
    for i, csv in enumerate(csvs):
        print(f"{i+1} out of {len(csvs)} done")
        year, month, day = csv.split(osSlash)[1][:4], csv.split(osSlash)[
            1][4:6], csv.split(osSlash)[1][6:8]
        # Check if day exists
        try:
            date = datetime.strptime(
                month + '/' + day + '/' + year, '%m/%d/%Y')
        except ValueError:
            continue
        df_uci = pd.read_csv(csv, usecols=["Time", 'Current demand'])
        df_uci['datetime'] = pd.to_datetime(
            month + '/' + day + '/' + year + ' ' + df_uci['Time'])
        df_uci = df_uci.drop(["Time"], axis=1)
        df_uci = df_uci.set_index('datetime')
        df_uci = df_uci.replace('?', np.nan)
        # df_uci = df_uci.astype(np.float).fillna(method='bfill')
        df = pd.concat([df, df_uci])
    df.to_csv("summedUp.csv")
else:
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
# df_uci_pivot = df_uci_pivot.dropna()

print(df_uci_pivot)
df_uci_pivot.T.plot(figsize=(13, 8), legend=False, color='blue', alpha=0.02)
plt.show()
