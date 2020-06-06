#imports
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from keras.models import load_model

#Global Variables
DATA_FOLDER_PATH = os.path.normpath(r'./data/')

# Data preprocessing
df = pd.read_csv(os.path.join(DATA_FOLDER_PATH, 'india.csv'))
training_set = df[df['status'] == 'confirmed']['cases'].values
training_set = training_set.reshape(-1, 1)

# Feauture scaling

sc = MinMaxScaler(feature_range=(0, 1))
scaled_training_set = sc.fit_transform(training_set)

# Creating Data Structure with 60 time steps and 1 output
x_train = []
y_train = []

for i in range(10, len(scaled_training_set)):
    x_train.append(scaled_training_set[i-10:i, 0])
    y_train.append(scaled_training_set[i, 0])

x_train = np.array(x_train)
y_train = np.array(y_train)

# Reshaping for other indicators
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)

# Building the RNN

# Initializing the RNN
regressor = Sequential()
# Building the layers
regressor.add(LSTM(units=50, return_sequences=True,
                   input_shape=(x_train.shape[1], 1)))
regressor.add(Dropout(rate=0.2))
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(rate=0.2))
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(rate=0.2))
regressor.add(LSTM(units=50, return_sequences=False))
regressor.add(Dropout(rate=0.2))
regressor.add(Dense(units=1))

# Compiling the RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')

#Training and Fitting
regressor.fit(x_train, y_train, epochs=100, batch_size=32)

#Loading model trained from Kaggle
#reg_pre = load_model('models\keras-basic-rnn-1')
reg_pre = regressor
#Prediction and Visualization

#Getting Test Data
test_df = pd.read_csv(os.path.join(DATA_FOLDER_PATH, 'germany.csv'))
test_set = test_df[test_df['status'] == 'confirmed']['cases'].values
y_actual = test_set.reshape(-1, 1)

#Preparing new data with 60 previous values for testing
#Getting full data
#df_total = pd.concat((df, test_df), axis=0, ignore_index=True)
#test_data = df_total.Open.values[-80:].reshape(-1, 1)
#Applying Scaling
test_data = y_actual[:]
inputs = sc.transform(test_data)
#Creating (60:1) format
x_test = []
for i in range(10, inputs.size):
    x_test.append(inputs[i-10:i, 0])
x_test = np.array(x_test)
#Reshaping to proper prediction structure
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

#Making predictions
y_pred = reg_pre.predict(x_test)

#Visualizing
y_pred = sc.inverse_transform(y_pred)
plt.plot(y_actual,'g', label='Actual Reported')
plt.plot(y_pred, 'r--', label='Predicted Reported')
plt.legend(loc='best')
plt.xlabel('Time')
plt.ylabel('Cases')
plt.title('Google Stock Price Prediction')
plt.show()