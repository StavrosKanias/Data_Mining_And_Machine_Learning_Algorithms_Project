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
        mean = dayMeanDemand(dfDemand)
        days.append(date)
        means.append(mean)
    ourDf = pandas.DataFrame({"Day": days, "Mean": means})

    plotdf(ourDf, "Monthly demand")


def plotdf(df, title):
    days = []
    means = []
    total = []
    prevyear = '2019'
    plt.figure(0)
    cnt = 0
    for day, mean in zip(df["Day"], df["Mean"]):
        year = str(day)[:4]
        if prevyear == year:
            days.append(day)
            means.append(mean)
        else:
            cnt += 1
            prevyear = year
            plt.subplot(3, 1, cnt)
            plt.title(f"{title} for the year {(int(prevyear) - 1)}")
            plt.plot_date(days, means, "b-", xdate=True)
            plt.plot_date(days, means, "r.", xdate=True)
            total.append((list(days), list(means)))
            days.clear()
            means.clear()
            days.append(day)
            means.append(mean)
    plt.subplot(3, 1, 3)
    plt.title(f"{title} for the year {(int(prevyear))}")
    plt.plot_date(days, means, "b-", xdate=True)
    plt.plot_date(days, means, "r.", xdate=True)
    plt.show()


def dayMeanSource(df):
    keys = list(df.columns.values)
    base = len(df[keys[-1]])


def dayMeanDemand(df):
    keys = list(df.columns.values)
    base = len(df[keys[-1]])
    total = 0
    cnt = 0
    for value in df[keys[-1]]:
        if math.isnan(value):
            cnt += 1
            continue
        total += value
    return total / (base - cnt)


if __name__ == "__main__":
    main()
