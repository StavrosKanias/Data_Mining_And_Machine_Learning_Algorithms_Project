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
        # print('Invalid date: ' + month + '/' + day + '/' + year)
        return year, month, day, False


def totalSupply(csv):
    csv['Total supply'] = 0
    for col in csv.columns:
        if col != 'Total supply':
            csv['Total supply'] += csv[col]
    return csv['Total supply']


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

                currentSupply = pd.read_csv(
                    supply[i]).drop(labels=["Time"], axis=1)
                currentSupply = currentSupply.replace(
                    '', np.nan).astype(np.float64).fillna(value=0.0)
                unified = pd.DataFrame(
                    columns=['Datetime', 'Demand', 'Supply'])
                # Datetime column
                unified['Datetime'] = pd.to_datetime(
                    month + '/' + day + '/' + year + ' ' + currentDemand['Time'])
                # Demand column
                unified["Demand"] = currentDemand['Current demand']
                # Supply column
                unified["Supply"] = totalSupply(currentSupply)

                unified = unified.set_index('Datetime')
                unified = unified.replace('?', np.nan).astype(
                    np.float64).fillna(method='bfill').dropna()
                # print(unified)
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


# # For simplication,
# # I will resample so that each row
# # represents a whole hour
df = unifyData()
df.plot()

df_uci_hourly = df.resample('H').sum()
df_uci_hourly['Hour'] = df_uci_hourly.index.hour
df_uci_hourly.index = df_uci_hourly.index.date

print(df_uci_hourly)

demand_pivot = df_uci_hourly.pivot(columns='Hour', values='Demand')
demand_pivot = demand_pivot.dropna()
supply_pivot = df_uci_hourly.pivot(columns='Hour', values='Supply')
supply_pivot = supply_pivot.dropna()

demand_pivot.T.plot(figsize=(13, 8), legend=False,
                    color='blue', alpha=0.02, title='Demand pivot')
supply_pivot.T.plot(figsize=(13, 8), legend=False,
                    color='red', alpha=0.02, title='Source pivot')
plt.show()
