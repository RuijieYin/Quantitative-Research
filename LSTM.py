# a basic LSTM model: predicting stock prices
# use Google (Alphabet) as an example

# use data downloaded from Yahoo Finance
import yfinance as yf
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

stock_data = yf.download('GOOGL', start='2015-01-01', end='2019-12-31')
# check the data:
stock_data.head()

# overview the data by plotting:
plt.plot(stock_data['Close'])

# split training and test data: 80% training and 20% test:
# note only close prices are used:
close_prices = stock_data['Close']
values = close_prices.values
training_data_len = math.ceil(len(values) * 0.8)

# will also need to normalize the data (range from 0 to 1)
# training data:
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(values.reshape(-1, 1))
train_data = scaled_data[0: training_data_len, :]

x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i - 60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# test data:
test_data = scaled_data[training_data_len - 60:, :]
x_test = []
y_test = values[training_data_len:]

for i in range(60, len(test_data)):
    x_test.append(test_data[i - 60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Model:
# use tensorflow to set up LSTM:
model = keras.Sequential()
model.add(layers.LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(layers.LSTM(100, return_sequences=False))
model.add(layers.Dense(25))
model.add(layers.Dense(1))
# see the model:
model.summary()

# train the model:
# use 'adam'
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=1, epochs=3)

# evaluation of the model on the test data:
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
rmse_results = np.sqrt(np.mean(predictions - y_test) ** 2)
# see the performance results:
rmse_results

# we can also visualize the predicted results:
data = stock_data.filter(['Close'])
train = data[:training_data_len]
validation = data[training_data_len:]
validation['Predictions'] = predictions
plt.figure(figsize=(15, 7))
plt.xlabel('Date')
plt.ylabel('Close Prices')
plt.plot(train)
plt.plot(validation[['Close', 'Predictions']])
plt.show()
