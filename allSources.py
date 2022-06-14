import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
from datetime import datetime


data_path = 'sources/*'
df = pd.DataFrame()

csvs = glob.glob(data_path)
csvs.sort()
osSlash = csvs[0][7]
if len(glob.glob("summedSources.csv")) == 0:
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
        df_temp = pd.read_csv(csv)
        cols = df_temp.columns.tolist()
        cols.remove("Solar")
        cols.remove("Wind")

        df_renewable = pd.read_csv(csv, usecols=["Time", 'Solar', 'Wind'])
        df_renewable['datetime'] = pd.to_datetime(
            month + '/' + day + '/' + year + ' ' + df_renewable['Time'])
        df_renewable = df_renewable.drop(["Time"], axis=1)
        df_renewable = df_renewable.set_index('datetime')
        df_renewable = df_renewable.replace('?', np.nan)
        renew = ["Solar", "Wind"]
        df_renewable['Renewable'] = df_renewable[renew].sum(axis=1)
        # df_uci = df_uci.astype(np.float).fillna(method='bfill')

        df_not_nenewable = pd.read_csv(csv, usecols=cols)
        df_not_nenewable['datetime'] = pd.to_datetime(
            month + '/' + day + '/' + year + ' ' + df_not_nenewable['Time'])
        df_not_nenewable = df_not_nenewable.drop(["Time"], axis=1)
        df_not_nenewable = df_not_nenewable.set_index('datetime')
        df_not_nenewable = df_not_nenewable.replace('?', np.nan)
        cols.remove("Time")
        df_not_nenewable["Non-Renewable"] = df_not_nenewable[cols].sum(axis=1)
        df_not_nenewable.drop(cols, axis=1, inplace=True)
        df_renewable.drop(renew, axis=1, inplace=True)
        df_temp = df_renewable.join(df_not_nenewable)
        df = pd.concat([df, df_temp])

    df["Total-Energy"] = df[["Renewable", "Non-Renewable"]].sum(axis=1)
    df.to_csv("summedSources.csv")
else:
    df = pd.read_csv("summedSources.csv")
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')

df = df.astype(np.float64).fillna(method='bfill')

df_uci_daily = df.resample('d').sum()
df_uci_daily['day'] = df_uci_daily.index.day
df_uci_daily.index = df_uci_daily.index.date
df_uci_daily.drop(['day'], axis=1, inplace=True)
print(df_uci_daily)
