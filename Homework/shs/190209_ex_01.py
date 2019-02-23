import numpy as np
import pandas as pd
from keras.layers import Dense,Activation
from keras.models import Sequential
from keras.datasets import mnist

x = np.array([[1,2,3,4,5,6,7,8,9,10],[101,102,103,104,105,106,107,108,109, 110]])
y = np.array([[1,2,3,4,5,6,7,8,9,10], [101,102,103,104,105,106,107,108,109,110]])

x = x.reshape(-1)
y = y.reshape(-1)

print(x.shape)
print(type(x))

x_train = x[:15]
y_train = y[:15]
x_test = x[15:]
y_test =y[15:]

# print(x_train)
# print(x_test)

# 모델 구성
model = Sequential()
# model.add(Dense(10, input_dim = 1, activation='relu'))
model.add(Dense(10, input_shape=(1,), activation='relu'))

model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
# model.compile(loss='mse', optimizer='adam', metrics=['acc'])

model.summary()

model.fit(x_train, y_train, epochs=200, batch_size=1, validation_data=(x_test, y_test))

a, b = model.evaluate(x_test, y_test, batch_size=1)
print(a, b)

y_predict = model.predict(x_test)
print(y_predict)


# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))