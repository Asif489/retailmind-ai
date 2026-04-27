import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def prepare_lstm(sales, window=3):
    scaler = MinMaxScaler()
    data = scaler.fit_transform(sales[['SALES']])

    X, y = [], []

    for i in range(len(data)-window):
        X.append(data[i:i+window])
        y.append(data[i+window])

    return np.array(X), np.array(y), scaler

def build_lstm(input_shape):
    model = Sequential([
        LSTM(64, activation='relu', input_shape=input_shape),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    return model