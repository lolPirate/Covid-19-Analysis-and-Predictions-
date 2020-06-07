from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('bmh')

# Global Variables
DATA_FOLDER_PATH = os.path.normpath(r'./data/')
MODEL_FOLDER_PATH = os.path.normpath(r'./models/')
MODEL_NAME = 'covid-predictor-keras-prtotype-version-001.h5'
TRAIN_DATA_FILES = ['india.csv', 'china.csv', 'iran.csv', 'australia.csv', 'canada.csv', 'italy.csv']


def get_training_data(country, status='confirmed'):
    '''Function to load training data from csv files'''
    df = pd.read_csv(os.path.join(DATA_FOLDER_PATH, country))
    train_data = df[df['status'] == status]['cases'].values
    train_data = train_data.reshape(-1, 1)
    return train_data


def create_model():
    '''Function to create a LSTM model'''
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(10, 1)))
    model.add(Dropout(rate=0.2))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(rate=0.2))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(rate=0.2))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(rate=0.2))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(rate=0.2))
    model.add(LSTM(units=100, return_sequences=False))
    model.add(Dropout(rate=0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def preprocess_data(train_data):
    '''Function to scale and process data'''
    sc = MinMaxScaler(feature_range=(0, 1))
    scaled_training_set = sc.fit_transform(train_data)

    x_train = []
    y_train = []

    for i in range(10, len(scaled_training_set)):
        x_train.append(scaled_training_set[i-10:i, 0])
        y_train.append(scaled_training_set[i, 0])

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)

    return x_train, y_train


def train(model):
    for country in TRAIN_DATA_FILES:
        print(f'Training on {country}')
        train_data = get_training_data(country)
        x_train, y_train = preprocess_data(train_data)
        model.fit(x_train, y_train, epochs=100, batch_size=32)
    return model


def predict(country, model):
    model = load_model(os.path.join(MODEL_FOLDER_PATH, MODEL_NAME))
    test_df = pd.read_csv(os.path.join(DATA_FOLDER_PATH, country))
    test_set = test_df[test_df['status'] == 'confirmed']['cases'].values
    y_actual = test_set.reshape(-1, 1)
    test_data = y_actual[:]
    sc = MinMaxScaler(feature_range=(0, 1))
    inputs = sc.fit_transform(test_data)
    x_test = []
    for i in range(10, inputs.size):
        x_test.append(inputs[i-10:i, 0])
    x_test = np.array(x_test)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
    y_pred = model.predict(x_test)
    y_pred = sc.inverse_transform(y_pred)
    
    y_pred = np.append(y_actual[0:10], y_pred)

    plt.plot(y_actual, 'g', label='Actual Reported')
    plt.plot(y_pred, 'r--', label='Predicted Reported')
    plt.legend(loc='best')
    plt.xlabel('Time')
    plt.ylabel('Cases')
    plt.title('Predicted vs Reported')
    plt.show()


if __name__ == '__main__':
    model = create_model()
    model = train(model)
    model.save(os.path.join(MODEL_FOLDER_PATH, MODEL_NAME))
    predict('spain.csv', MODEL_NAME)
