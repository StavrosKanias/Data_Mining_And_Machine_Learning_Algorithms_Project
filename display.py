import pandas
import glob
from datetime import datetime
import math
import matplotlib.pyplot as plt
import random

osSlash = "/"


def main():
    global osSlash
    demands = glob.glob("demand/*.csv")
    sources = glob.glob("sources/*.csv")
    sources.sort()
    demands.sort()
    osSlash = demands[0][6]
    days = []
    means = []
    expectedMeans = []
    energy = []
    for demand, source in zip(demands, sources):
        year, month, day = demand.split(osSlash)[1][:4], demand.split(osSlash)[
            1][4:6], demand.split(osSlash)[1][6:8]
        # Check if day exists
        try:
            date = datetime.strptime(
                month + '/' + day + '/' + year, '%m/%d/%Y')
        except ValueError:
            continue

        # Read csv
        dfDemand = pandas.read_csv(demand)
        dfSource = pandas.read_csv(source)
        energy.append(dayMeanSource(dfSource))
        mean = dayMeanDemand(dfDemand, "Current demand")
        expectedMean = dayMeanDemand(dfDemand, "Day ahead forecast")
        days.append(date)
        means.append(mean)
        expectedMeans.append(expectedMean)
    # Επειδή δεν έχουμε τιμή για την πρώτη μέρα βάζουμε την ίδια
    expectedMeans.insert(0, means[0])
    demandDf = pandas.DataFrame(
        {"Day": days, "Mean": means, "Day ahead forecast": expectedMeans[:-1], "Source": energy})
    showRandomDays()
    plotdf(demandDf, "Monthly demand", 0)
    plt.show()


def pieCharts():
    global osSlash
    dates = ("20190501", "20200914", "20211212", "20200906")
    normalDate = ("05/01/2019", "09/14/2020", "12/12/2021", "09/06/2020")
    for id, date in enumerate(dates):
        # plt.figure(id+1)
        cursource = "sources"+osSlash+date+".csv"
        dfSource = pandas.read_csv(cursource)
        dfSource.drop(["Time"], axis=1, inplace=True)
        scores = pandas.DataFrame(
            {'team': dfSource.columns.to_list(), 'production': [dfSource[i].sum() for i in dfSource.columns]})
        scores.drop(scores[scores['production'] <= 0].index, inplace=True)
        scores.groupby(['team']).sum().plot(
            kind='pie', y='production', autopct='%1.2f%%', title=normalDate[id])


def yearPieChart(year="2019"):
    df = pandas.read_csv("unified.csv")
    df['Datetime'] = [i.split('-')[0] for i in df['Datetime']]
    df = df[df.Datetime == year]
    df.drop(['Datetime', 'Demand', 'Supply', "Renewable",
            'Non-Renewable'], axis=1, inplace=True)
    scores = pandas.DataFrame({'team': df.columns.to_list(), 'production': [
                              df[i].sum() for i in df.columns]})
    scores.drop(scores[scores['production'] <= 0].index, inplace=True)
    scores.groupby(['team']).sum().plot(
        kind='pie', y='production', autopct='%1.2f%%', title=f"PieChart for year {year}")


def histPlot():
    global osSlash
    dates = ("20190501", "20200914", "20211212", "20200906")
    normalDate = ("05/01/2019", "09/14/2020", "12/12/2021", "09/06/2020")
    for id, date in enumerate(dates):
        # plt.figure(id+1)
        cursource = "sources"+osSlash+date+".csv"
        dfSource = pandas.read_csv(cursource)
        dfSource.drop(["Time"], axis=1, inplace=True)
        colors = []
        for _ in dfSource.columns.tolist():
            rgb = (random.random(), random.random(), random.random())
            colors.append(rgb)
        scores = pandas.DataFrame(
            {'team': dfSource.columns.to_list(), 'production': [dfSource[i].sum() for i in dfSource.columns]})
        scores.plot.barh(x='team', y='production',
                         title=normalDate[id], color=colors)
        plt.savefig('plots/production' + date)


def yearHistPlot(year="2019"):
    df = pandas.read_csv("unified.csv")
    df['Datetime'] = [i.split('-')[0] for i in df['Datetime']]
    df = df[df.Datetime == year]
    df.drop(['Datetime', 'Demand', 'Supply', "Renewable",
            'Non-Renewable'], axis=1, inplace=True)
    colors = []
    for _ in df.columns.tolist():
        rgb = (random.random(), random.random(), random.random())
        colors.append(rgb)
    scores = pandas.DataFrame({'team': df.columns.to_list(), 'production': [
                              df[i].sum() for i in df.columns]})
    scores.plot.barh(x='team', y='production',
                     title=f'Year = {year}', color=colors)
    plt.savefig('plots/production' + year)


def categorizeEnergy(df):
    time = pandas.DataFrame(df[["Time"]])
    renewable = pandas.DataFrame(df[["Solar", "Wind"]])
    nonRenewable = pandas.DataFrame(df[df.columns[3:]])
    return time, renewable, nonRenewable


def showRandomDays():
    global osSlash
    dates = ("20190501", "20200914", "20211212", "20200906")
    for id, date in enumerate(dates):
        plt.figure(id+1)
        curdemand = "demand"+osSlash+date+".csv"
        cursource = "sources"+osSlash+date+".csv"
        dfDemand = pandas.read_csv(curdemand)
        dfSource = pandas.read_csv(cursource)
        duck_curve(dfDemand, dfSource, date)


def duck_curve(dfDemand, dfSource, day):
    time, renewable, nonRenewable = categorizeEnergy(dfSource)
    renewable["SUM"] = renewable.sum(axis=1)
    nonRenewable["SUM"] = nonRenewable.sum(axis=1)
    currentDemand = pandas.DataFrame(dfDemand[["Current demand"]][: -1])
    plt.title(day)
    plt.plot(time["Time"], currentDemand["Current demand"],
             label="Current Demand")
    plt.plot(time["Time"], renewable["SUM"], label="Non-Dispatchable")
    plt.plot(time["Time"], nonRenewable["SUM"], label="Dispatchable")
    plt.grid()
    plt.legend()


def plotdf(demandDf, title, figure):
    days = []
    means = []
    total = []
    expectedDemand = []
    energy = []
    prevyear = '2019'
    plt.figure(figure)
    cnt = 0
    for day, mean, expected, produced in zip(demandDf["Day"], demandDf["Mean"], demandDf["Day ahead forecast"], demandDf["Source"]):
        year = str(day)[: 4]
        if prevyear == year:
            days.append(day)
            means.append(mean)
            expectedDemand.append(expected)
            energy.append(produced)
        else:
            cnt += 1
            prevyear = year
            plt.subplot(3, 1, cnt)
            plt.title(f"{title} for the year {(int(prevyear) - 1)}")
            # plt.plot_date(days, means, "b-", xdate=True)
            plt.plot_date(days, means, "r-", xdate=True)
            plt.plot_date(days, expectedDemand, "b-", xdate=True)
            plt.plot_date(days, energy, "y-", xdate=True)
            plt.plot_date(days, energy, "y.", xdate=True, markersize=3)
            plt.plot_date(days, means, "r.", xdate=True, markersize=3)
            plt.plot_date(days, expectedDemand, "b.",
                          xdate=True, markersize=3)
            plt.legend(("Day Mean", "Day Predicted", "Energy Produced"))
            total.append((list(days), list(means)))
            days.clear()
            means.clear()
            energy.clear()
            expectedDemand.clear()
            energy.append(produced)
            expectedDemand.append(expected)
            days.append(day)
            means.append(mean)
    plt.subplot(3, 1, 3)
    plt.title(f"{title} for the year {(int(prevyear))}")
    plt.plot_date(days, means, "r-", xdate=True)
    plt.plot_date(days, expectedDemand, "b-", xdate=True)
    plt.plot_date(days, energy, "y-", xdate=True)
    plt.plot_date(days, energy, "y.", xdate=True, markersize=3)
    plt.plot_date(days, means, "r.", xdate=True, markersize=3)
    plt.plot_date(days, expectedDemand, "b.", xdate=True, markersize=3)
    plt.legend(("Day Mean", "Day Predicted", "Energy Produced"))


def dayMeanSource(df):
    keys = list(df.columns.values)
    keys.remove("Time")
    totalEnergy = 0
    base = len(df[keys[0]])
    for key in keys:
        for value in df[key]:
            if math.isnan(value):
                continue
            totalEnergy += value
    return totalEnergy / base


def dayMeanDemand(df, column):
    keys = list(df.columns.values)
    if column in keys:
        base = len(df[column])
        total = 0
        cnt = 0
        for value in df[column]:
            if math.isnan(value):
                cnt += 1
                continue
            total += value
        return total / (base - cnt)


if __name__ == "__main__":
    # main()
    pieCharts()
    yearPieChart('2019')
    histPlot()
    yearHistPlot('2019')
    plt.show()
