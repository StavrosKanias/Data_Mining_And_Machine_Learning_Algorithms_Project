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
        # dfSource = pandas.read_csv(source)
        mean = dayMeanDemand(dfDemand, "Current demand")
        expectedMean = dayMeanDemand(dfDemand, "Day ahead forecast")
        days.append(date)
        means.append(mean)
        expectedMeans.append(expectedMean)
    demandDf = pandas.DataFrame({"Day": days, "Mean": means})
    expectedDemandDf = pandas.DataFrame(
        {"Day": days, "Mean": expectedMeans})

    plotdf(demandDf, expectedDemandDf, "Monthly demand", 0)
    plt.show()


def plotdf(demandDf, expectedDemandDf, title, figure):
    days = []
    means = []
    total = []
    expectedDemand = []
    prevyear = '2019'
    plt.figure(figure)
    cnt = 0
    for day, mean, expected in zip(demandDf["Day"], demandDf["Mean"], expectedDemandDf["Mean"]):
        year = str(day)[:4]
        if prevyear == year:
            days.append(day)
            means.append(mean)
            expectedDemand.append(expected)
        else:
            cnt += 1
            prevyear = year
            plt.subplot(3, 1, cnt)
            plt.title(f"{title} for the year {(int(prevyear) - 1)}")
            # plt.plot_date(days, means, "b-", xdate=True)
            plt.plot_date(days, means, "rx", xdate=True)
            ###
            temp = expectedDemand.pop(-1)
            ###
            plt.plot_date(days[1 if cnt == 1 else 0:],
                          expectedDemand, "b.", xdate=True)

            total.append((list(days), list(means)))
            days.clear()
            means.clear()
            expectedDemand.clear()
            expectedDemand.extend((temp, expected))
            days.append(day)
            means.append(mean)
    plt.subplot(3, 1, 3)
    plt.title(f"{title} for the year {(int(prevyear))}")
    plt.plot_date(days, expectedDemand[:-1], "rx", xdate=True)
    plt.plot_date(days, means, "b.", xdate=True)


def dayMeanSource(df):
    keys = list(df.columns.values)
    base = len(df[keys[-1]])


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
