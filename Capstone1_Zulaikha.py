# -*- coding: utf-8 -*-
"""Untitled31.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1F965iUBjSEBQfTI9IRSU7nHMTj36yPNQ
"""

# Commented out IPython magic to ensure Python compatibility.
## 1. Import packages
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
# %matplotlib inline

import os
print(os.getcwd())

## 2. Data loading
PATH = os.getcwd()
TRAIN_PATH = os.path.join(PATH, "/content/cases_malaysia_train.csv")
TEST_PATH = os.path.join(PATH, "/content/cases_malaysia_test.csv")
df_train = pd.read_csv(TRAIN_PATH)
df_test = pd.read_csv(TEST_PATH)

## 3. Data Inspection
# (A) check for info
print(" -------------------TRAIN DATA INFO----------------")
df_train.info()

#(B) check for info
print(" -------------------TEST DATA INFO----------------")
df_test.info()

# (B) Check fo NA values
df_train.isna().sum()

## 4. Data Cleaning
df_train["cluster_import"].fillna(df_train["cluster_import"].mean(), inplace=True)
df_train["cluster_highRisk"].fillna(df_train["cluster_highRisk"].median(), inplace=True)
df_train["cluster_education"].fillna(df_train["cluster_education"].median(), inplace=True)
df_train["cluster_detentionCentre"].fillna(df_train["cluster_detentionCentre"].median(), inplace=True)
df_train["cluster_workplace"].fillna(df_train["cluster_workplace"].median(), inplace=True)

# Fill up NA values
df_train["cases_new"] = df_train["cases_new"].interpolate()
df_train.isna().sum()

print(df_train.columns)

print(df_test.columns)

non_numeric_rows = df_train[(df_train['cases_new'] == '?') | pd.to_numeric(df_train['cases_new'], errors='coerce').isna()]
print(non_numeric_rows)

df_train['cases_new'] = df_train['cases_new'].replace('?', np.nan)
df_train['cases_new'] = pd.to_numeric(df_train['cases_new'], errors='coerce')

print(df_train['cases_new'])

df_train['cases_new'] = df_train['cases_new'].interpolate()

df_train['cases_new'] = df_train['cases_new'].astype('int64')
print(df_train['cases_new'])

df_test['cases_new'] = df_test['cases_new'].interpolate()
df_test['cases_new'] = df_test['cases_new'].astype('int64')

## 5. Data Preprocessing
from sklearn.preprocessing import MinMaxScaler

df_train_cases = df_train['cases_new']
df_test_cases = df_test['cases_new']

mms = MinMaxScaler()
df_train_cases_scaled = mms.fit_transform(np.expand_dims(df_train_cases,axis=-1))
df_test_cases_scaled = mms.transform(np.expand_dims(df_test_cases,axis=-1))

#Data windowing
window_size = 30 #30 window = 30 input
X_train = []
y_train = []

for i in range(window_size,len(df_train_cases_scaled)):
    X_train.append(df_train_cases_scaled[i-window_size:i])
    y_train.append(df_train_cases_scaled[i,0])

X_train = np.array(X_train)
y_train = np.array(y_train)

#Concatenate train and test data together
df_cases_stacked = np.concatenate((df_train_cases_scaled,df_test_cases_scaled))

#use method 2
length_days = window_size + len(df_test_cases_scaled)
tot_input = df_cases_stacked[-length_days:]
data_test = df_cases_stacked[-length_days:]

X_test = []
y_test = []

for i in range(window_size, len(data_test)):
    X_test.append(data_test[i-window_size:i])
    y_test.append(data_test[i])

X_train = np.array(X_train)
y_train = np.array(y_train)

## 8. Model development
from tensorflow.keras import Sequential,Input
from tensorflow.keras.layers import LSTM,Dropout,Dense
from tensorflow.keras.utils import plot_model

input_shape = np.shape(X_train)[1:]

model = Sequential()
model.add(Input(shape=(input_shape)))
model.add(LSTM(64,return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64,return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64))
model.add(Dropout(0.3))
model.add(Dense(1)) 

model.summary()
plot_model(model,show_shapes=True,show_layer_names=True)

## 9. Model Compilation
model.compile(optimizer='adam',loss='mse',metrics=['mae','mape','mse'])

## 10. Model Training
print(X_train.shape)
print(y_train.shape) 
y_train = y_train.reshape((y_train.shape[0], 1))

X_test = []
y_test = []

for i in range(window_size, len(data_test)):
    X_test.append(data_test[i - window_size:i])
    y_test.append(data_test[i])

X_test = np.array(X_test)
y_test = np.array(y_test)

#Reshape X_test to have the same shape as X_train
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

#Create a TensorBoard callback object for the usage of TensorBoard
import tensorflow as tf
import datetime
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import TensorBoard
base_log_path = r"tensorboard_logs\ass2"
log_path = os.path.join(base_log_path,datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb = callbacks.TensorBoard(log_path)

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, callbacks=[tb])

# To check the MAPE error

# Obtain predictions on the testing dataset
predictions = model.predict(X_test)

# Calculate APE for each data point
ape = np.abs((y_test - predictions) / y_test) * 100

# Calculate MAPE for the testing dataset
mape = np.mean(ape)

# Calculate MAE for the testing dataset
mae = np.mean(np.abs(y_test - predictions))

# Calculate MABE for the testing dataset
mabe = np.mean(np.abs(y_test - np.mean(y_test)))

# Calculate MAPE Error Ratio
mape_error_ratio = ((mae - mabe) / mabe) * 100

# Check if MAPE Error Ratio is less than 1%
if mape_error_ratio < 1:
    print("MAPE is less than 1 percent for the testing dataset.")
else:
    print("MAPE is not less than 1 percent for the testing dataset.")

## 11. Model Evaluation

print(history.history.keys())

#Plot the evaluation graph
plt.figure()
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.legend(['Train MAE','Test MAE'])
plt.show()

## 12. Model Deployment
y_pred = model.predict(X_test)

#perform inverse transform
actual_cases = mms.inverse_transform(y_test)
predicted_cases = mms.inverse_transform(y_pred)

#Plot actual vs predicted
plt.figure()
plt.plot(actual_cases,color='red')
plt.plot(predicted_cases,color='blue')
plt.xlabel("Days")
plt.ylabel("New Cases in Malaysia)")
plt.legend(['Actual','Predicted'])