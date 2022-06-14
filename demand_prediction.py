import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
np.random.seed(7)

outliers = [
    "2019-01-02",
    "2019-03-10",
    "2019-03-13",
    "2019-03-26",
    "2019-05-27",
    "2019-06-11",
    "2019-07-23",
    "2019-07-24",
    "2019-07-25",
    "2019-07-26",
    "2019-08-14",
    "2019-08-15",
    "2019-08-26",
    "2019-08-27",
    "2019-09-03",
    "2019-09-04",
    "2019-09-05",
    "2019-09-06",
    "2020-03-29",
    "2020-04-11",
    "2020-04-12",
    "2020-04-19",
    "2020-08-13",
    "2020-08-14",
    "2020-08-15",
    "2020-08-16",
    "2020-08-17",
    "2020-08-18",
    "2020-08-19",
    "2020-08-20",
    "2020-08-21",
    "2020-08-22",
    "2020-08-24",
    "2020-08-25",
    "2020-09-05",
    "2020-09-06",
    "2020-09-07",
    "2021-06-18",
    "2021-07-09",
    "2021-07-19",
    "2021-07-20",
    "2021-07-21",
    "2021-07-29",
    "2021-08-11",
    "2021-08-12",
    "2021-08-16",
    "2021-09-08",
    "2021-09-09",
    "2021-12-14",
    "2021-12-24",
    "2021-12-25",
    "2021-12-26",
    "2021-12-30"
]


# convay of values into a dataset matrix


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


def read_dataset():
    df_demand = pd.read_csv("unified.csv")
    df_sources = pd.read_csv("summedSources.csv")
    df_demand.drop('Supply', axis=1)
    df_demand.drop('Datetime', axis=1)
    df = df_sources.join(df_demand)
    df['Demand-Renewable'] = df['Demand'] - df['Renewable']
    # df = df.iloc[:110000, :]
    df = df.dropna()

    for i in outliers:
        curIndex = df.loc[df['datetime'] == f'{i} 00:00:00'].index
        print(curIndex)
        for j in range(288):
            df.drop(curIndex + j)
    return df


if __name__ == "__main__":
    df = read_dataset()
    print(df)
    cols = ['Demand-Renewable']
    dataframe = df[cols]
    dataset = dataframe.values
    dataset = dataset.astype('float32')
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    # split into train and test sets
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    # print(len(train), len(test))
    look_back = 4
    trainX, trainY = create_dataset(train, look_back=look_back)
    testX, testY = create_dataset(test, look_back=look_back)
    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(32, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=5, batch_size=128)

    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
    print('Test Score: %.2f RMSE' % (testScore))

    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict)+(look_back*2) +
                    1:len(dataset)-1, :] = testPredict
    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(dataset))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()
