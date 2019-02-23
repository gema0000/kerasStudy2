#필요한 모듈 불러오기
import numpy as np
import pandas as pd
from keras.layers import Dense,Activation
from keras.models import Sequential
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

# train set build
x=np.array([range(1,11),range(101,111)])
x=np.transpose(x)
y=np.array([range(1,11)])
y=np.transpose(y)

#model build
model=Sequential()
model.add(Dense(10, input_shape=(2,), activation='relu',kernel_initializer='he_uniform'))
model.add(Dense(10))
model.add(Dense(10, activation='relu',kernel_initializer='he_uniform'))
model.add(Dense(1))
model.compile(loss='mse',optimizer='adamax',metrics=['mse'])

#show model structure
model.summary()

#train model
model.fit(x,y,epochs=300,batch_size=1)

#predict x 
a=model.predict(x)

# get r2 score
r2_y_predict = r2_score(y, a)
print("R2 : ", r2_y_predict)

# get rmse 
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y, a))
