import pandas
import glob
from datetime import datetime
import math
import matplotlib.pyplot as plt


def main():
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
    plotdf(demandDf, "Monthly demand", 0)
    plt.show()


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
        year = str(day)[:4]
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
    main()
