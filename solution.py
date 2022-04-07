import pandas
import glob
from datetime import datetime
import math


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

        name = file.split(osSlash)[1]
        # Read csv
        df = pandas.read_csv(file)
        mean = dayMeanValue(df)
        days.append(date)
        means.append(mean)
    ourDf = pandas.DataFrame({"Day": days, "Mean": means})
    print(ourDf)


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
