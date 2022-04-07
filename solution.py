import pandas
import glob
from datetime import datetime
import math
import matplotlib.pyplot as plt


def main():
    demand = glob.glob("demand/*.csv")
    demand.sort()
    osSlash = demand[0][6]
    days = []
    means = []
    for file in demand:
        year, month, day = file.split(osSlash)[1][:4], file.split(osSlash)[
            1][4:6], file.split(osSlash)[1][6:8]
        # Check if day exists
        try:
            date = datetime.strptime(
                month + '/' + day + '/' + year, '%m/%d/%Y')
        except ValueError:
            continue

        # Read csv
        df = pandas.read_csv(file)
        mean = dayMeanValue(df)
        days.append(date)
        means.append(mean)
    ourDf = pandas.DataFrame({"Day": days, "Mean": means})
    plotdf(ourDf)


def plotdf(df):
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
            plt.plot_date(days, means, "b-", xdate=True)
            plt.plot_date(days, means, "r.", xdate=True)
            total.append((list(days), list(means)))
            days.clear()
            means.clear()
            days.append(day)
            means.append(mean)
    plt.subplot(3, 1, 3)
    plt.plot_date(days, means, "b-", xdate=True)
    plt.plot_date(days, means, "r.", xdate=True)
    plt.show()


def dayMeanValue(df):
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
