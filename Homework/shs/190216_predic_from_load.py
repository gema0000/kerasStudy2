from keras.models import Sequential, Model
from keras.models import load_model
import numpy as np
from dataset import x1_test, x2_test, x3_test, y1_test, y2_test, y3_test, merXtest, merYtest

# 모델 불러오기
model = load_model('savedModelWeight.h5')

# y_predict = model.predict([x1_test, x2_test, x3_test])   # list
y_predict = model.predict(merXtest)   # list
print(y_predict)        

# y_predict = y_predict.value()
y_predict = np.array(y_predict)

print(type(y_predict))
print(y_predict.shape)

y_predict = y_predict.flatten()
print(y_predict)        

print(type(y_predict))
print(y_predict.shape)

print('끗')

# R2 구하기
from sklearn.metrics import r2_score
# r2_y_predict = r2_score((np.array([y1_test, y2_test, y3_test])).flatten(), y_predict)
r2_y_predict = r2_score((np.array(merYtest)).flatten(), y_predict)
# r2_y_predict = r2_score(y1_test + y2_test, y_predict)
print("R2 : ", r2_y_predict)    

# RMSE 구하기
from sklearn.metrics import mean_squared_error, mean_absolute_error
def RMSE(y12_test, y_predict):
    return np.sqrt(mean_squared_error(y12_test, y_predict))
# print("RMSE : ", RMSE((np.array([y1_test, y2_test, y3_test])).flatten(), y_predict))
print("RMSE : ", RMSE((np.array(merYtest)).flatten(), y_predict))


# loss = 0.0597
# R2 = 0.999