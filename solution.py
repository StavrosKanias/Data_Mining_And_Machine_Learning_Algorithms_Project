import pandas
import glob
from datetime import datetime


def main():
    demand = glob.glob("demand/*.csv")
    demand.sort()
    osSlash = demand[0][6]
    ourDf = pandas.DataFrame()
    for file in demand:
<<<<<<< HEAD
        year, month, day = file.split(
            '/')[1][:4], file.split('/')[1][4:6], file.split('/')[1][6:8]
=======
        year , month , day = file.split(osSlash)[1][:4] , file.split(osSlash)[1][4:6] , file.split(osSlash)[1][6:8]
>>>>>>> 27ffa0ea27a80c67ec2f4b827a32f18ed2c6edc8
        # print (f'{year=} {month=} {day=}')
        try:
            datetime.strptime(month + '/' + day + '/' + year, '%m/%d/%Y')
        except ValueError:
            print(file)
            continue
        name = file.split('/')[1]
        # print(file)
        df = pandas.read_csv(file)
    print(df)


if __name__ == "__main__":
    main()
