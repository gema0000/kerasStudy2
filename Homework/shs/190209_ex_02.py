import numpy as np
import pandas as pd
from keras.layers import Dense,Activation
from keras.backend import concatenate
from keras.models import Sequential
from keras.datasets import mnist

x_1 = np.array([1,2,3,4,5,6,7,8,9,10])
x_2 = np.array([101,102,103,104,105,106,107,108,109, 110])

y_1 = np.array([1,2,3,4,5,6,7,8,9,10])
y_2 = np.array([101,102,103,104,105,106,107,108,109,110])

mergedX = concatenate([x_1,x_2],axis=0)
mergedY = concatenate([y_1,y_2],axis=0)

# print(x.shape)
# print(type(x))

mergedX_train = mergedX[:15]
mergedY_train = mergedY[:15]
mergedX_test = mergedX[15:]
mergedY_test =mergedY[15:]

print(mergedX_train)
print(mergedX_test)

# 모델 구성
model = Sequential()
# model.add(Dense(10, input_dim = 1, activation='relu'))
model.add(Dense(10, input_shape=(1,), activation='relu'))

model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
# model.compile(loss='mse', optimizer='adam', metrics=['acc'])

model.summary()

model.fit(mergedX_train, mergedY_train, epochs=50, batch_size=None, steps_per_epoch = 1, validation_steps=1, validation_data=(mergedX_test, mergedY_test))

a, b = model.evaluate(mergedX_test, mergedY_test, batch_size=1)
print(a, b)

y_predict = model.predict(mergedX_test)
print(y_predict)


# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(mergedY_test, y_predict)
print("R2 : ", r2_y_predict)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(mergedY_test, y_predict):
    return np.sqrt(mean_squared_error(mergedY_test, y_predict))
print("RMSE : ", RMSE(mergedY_test, y_predict))