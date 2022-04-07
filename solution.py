import pandas
import glob
from datetime import datetime


def main():
    demand = glob.glob("demand/*.csv")
    demand.sort()
    osSlash = demand[0][6]
    ourDf = pandas.DataFrame()
    for file in demand:
        year, month, day = file.split(osSlash)[1][:4], file.split(osSlash)[
            1][4:6], file.split(osSlash)[1][6:8]
        # print (f'{year=} {month=} {day=}')
        try:
            datetime.strptime(month + '/' + day + '/' + year, '%m/%d/%Y')
        except ValueError:
            print(file)
            continue
        name = file.split(osSlash)[1]
        # print(file)
        df = pandas.read_csv(file)
    print(df)


if __name__ == "__main__":
    main()
