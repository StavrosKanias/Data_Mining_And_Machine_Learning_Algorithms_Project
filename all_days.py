import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
from datetime import datetime
from csv import reader


def hourly(df, column):
    hourlyMean = []
    temp = 0
    for i, demand in enumerate(df[column]):
        if i % 12 == 0 and i != 0:
            hourlyMean.append(temp)
            temp = 0
        temp += demand
    return hourlyMean


def validDate(csv, osSlash):
    year, month, day = csv.split(osSlash)[1][:4], csv.split(osSlash)[
        1][4:6], csv.split(osSlash)[1][6:8]
    # Check if date exists
    try:
        date = datetime.strptime(
            month + '/' + day + '/' + year, '%m/%d/%Y')
        return year, month, day, True
    except ValueError:
        return year, month, day, False


def getSupply(csv):
    currentSupply = pd.read_csv(
        csv).drop(labels=["Time"], axis=1)
    currentSupply = currentSupply.replace('', np.nan).astype(
        np.float64).fillna(value=0.0).dropna()
    cols = currentSupply.columns.tolist()

    renew = ["Solar", "Wind"]
    df_renewable = pd.read_csv(csv, usecols=renew).replace('', np.nan).astype(
        np.float64).fillna(method='bfill')
    df_renewable['Renewable'] = df_renewable[renew].sum(axis=1)

    cols.remove("Solar")
    cols.remove("Wind")
    df_not_renewable = pd.read_csv(csv, usecols=cols).replace('', np.nan).astype(
        np.float64).fillna(method='bfill')
    df_not_renewable = df_not_renewable.replace('', np.nan)
    for i, c in enumerate(cols):
        stdc = c[0].upper() + c[1:].lower()
        df_not_renewable.rename(columns={c: stdc}, inplace=True)
        cols[i] = stdc
    df_not_renewable["Non-Renewable"] = df_not_renewable[cols].sum(axis=1)

    dfOut = df_renewable.join(df_not_renewable)
    dfOut['Supply'] = totalSupply(currentSupply)
    allCols = ['Supply', 'Renewable', 'Non-Renewable'] + renew + cols
    dfOut = dfOut.reindex(columns=allCols)
    # print(dfOut)
    return dfOut


def totalSupply(dfIn):
    dfIn['Total supply'] = 0
    for col in dfIn.columns:
        if col != 'Total supply':
            dfIn['Total supply'] += dfIn[col]
    return dfIn['Total supply']


def unifyData():

    plt.style.use('seaborn')
    df = pd.DataFrame()
    demand_path = 'demand/*'
    demand = glob.glob(demand_path)
    demand.sort()
    supply_path = 'sources/*'
    supply = glob.glob(supply_path)
    supply.sort()

    if len(demand) != len(supply):
        print('Invalid data \n Demand data not equal in length with supply data')
        return
    else:
        csvs = len(demand)

    osSlash = demand[0][6]

    if len(glob.glob("unified.csv")) == 0:
        for i in range(csvs):
            year, month, day, valid = validDate(demand[i], osSlash)
            if(valid):

                currentDemand = pd.read_csv(
                    demand[i], usecols=['Time', 'Current demand'])
                total_supply = getSupply(supply[i])
                unified = pd.DataFrame()
                if total_supply.isnull().values.any():
                    print('Invalid data \n Supply data is null')
                    continue
                # Datetime column
                unified['Datetime'] = pd.to_datetime(
                    month + '/' + day + '/' + year + ' ' + currentDemand['Time'])
                # Demand column
                unified["Demand"] = currentDemand['Current demand']

                # Supply columns

                total_supply['Datetime'] = pd.to_datetime(
                    month + '/' + day + '/' + year + ' ' + currentDemand['Time'])

                unified = unified.set_index('Datetime')
                total_supply = total_supply.set_index('Datetime')

                unified = unified.replace('', np.nan).astype(
                    np.float64).fillna(method='bfill')
                total_supply = total_supply.replace('', np.nan).astype(
                    np.float64).fillna(method='bfill')
                # print(total_supply)
                unified = unified.join(total_supply)
                df = pd.concat([df, unified])
                print(f"{i+1} out of {csvs} included")

        df.to_csv("unified.csv")

    else:
        df = pd.read_csv("unified.csv")
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df = df.set_index('Datetime')

    df = df.astype(np.float64).fillna(method='bfill')
    print(df)
    return df


df = unifyData()
# df.plot()

# df_uci_hourly = df.resample('H').sum()
# df_uci_hourly['Hour'] = df_uci_hourly.index.hour
# df_uci_hourly.index = df_uci_hourly.index.date

# print(df_uci_hourly)

# demand_pivot = df_uci_hourly.pivot(columns='Hour', values='Demand')
# demand_pivot = demand_pivot.dropna()
# supply_pivot = df_uci_hourly.pivot(columns='Hour', values='Supply')
# supply_pivot = supply_pivot.dropna()

# demand_pivot.T.plot(figsize=(13, 8), legend=False,
#                     color='blue', alpha=0.02, title='Demand pivot')
# supply_pivot.T.plot(figsize=(13, 8), legend=False,
#                     color='red', alpha=0.02, title='Source pivot')
# plt.show()
