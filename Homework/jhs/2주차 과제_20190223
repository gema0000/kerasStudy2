#필요한 모듈 호출
import numpy as np
from keras.layers import Dense, Activation, Concatenate, Input
from keras.models import Model
from keras.layers.merge import concatenate
from keras.models import load_model
#tensorboard
from tensorflow.python.keras.callbacks import TensorBoard

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

#rmse함수 정의
def RMSE(y12_test, y_predict):
    return np.sqrt(mean_squared_error(y12_test, y_predict))

#데이터 셋 생성
x1 = np.array([1,2,3,4,5,6,7,8,9,10])
x2 = np.array([101,102,103,104,105,106,107,108,109,110])
x3=np.array([range(1001,1101)])
x3=x3.flatten()
y1 = np.array([1,2,3,4,5,6,7,8,9,10])
y2 = np.array([101,102,103,104,105,106,107,108,109,110])
y3=np.array([range(1001,1101)])
y3=y3.flatten()

#학습 데이터와 테스트 데이터 생성
x1_train = x1[:7]
x2_train = x2[:7]       
y1_train = y1[:7]
y2_train = y2[:7]
x3_train=x3[:70]
y3_train=y3[:70]
x1_test = x1[7:]  
x2_test = x2[7:]  
y1_test =y1[7:]  
y2_test =y2[7:]
x3_test=x3[70:]
y3_test=y3[70:]
x1_train=np.tile(x1_train,10)
x2_train=np.tile(x2_train,10)
y1_train=np.tile(y1_train,10)
y2_train=np.tile(y2_train,10)
x1_test=np.tile(x1_test,10)
x2_test=np.tile(x2_test,10)
y1_test=np.tile(y1_test,10)
y2_test=np.tile(y2_test,10)

# 모델 구성
# model 1
input1 = Input(shape=(1,))
dense1 = Dense(150, activation='relu')(input1)
# model 2
input2 = Input(shape=(1,))
dense2 = Dense(150, activation='relu')(input2)

#model 3
input3=Input(shape=(1,))
dense3=Dense(150,activation='relu')(input3)
# concat
merge1 = concatenate([dense1, dense2, dense3])
#split
output_1 = Dense(50)(merge1)
output1 = Dense(1)(output_1)
output_2 = Dense(50)(merge1)
output2 = Dense(1)(output_2)
output_3=Dense(50)(merge1)
output3=Dense(1)(output_3)

#모델 객체 생성
model = Model(inputs=[input1, input2, input3], outputs=[output1, output2, output3])
#모델 컴파일
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
#모델 구조
model.summary()
#텐서보드 디렉트리 설정
tensorboard=TensorBoard(log_dir="logs")
#텐서보드에 모델을 callbacks 객체로 할당?
model.fit([x1_train,x2_train,x3_train],[y1_train,y2_train,y3_train],epochs=1000,batch_size=16
          ,callbacks=[tensorboard])

y_predict=model.predict([x1_test,x2_test,x3_test])
y_predict=np.array(y_predict).flatten()
#r2 score
r2_y_predict = r2_score((np.array([y1_test, y2_test,y3_test])).flatten(), y_predict)
# r2_y_predict = r2_score(y1_test + y2_test, y_predict)
print("R2 : ", r2_y_predict)
#RMSE score
print("RMSE : ", RMSE((np.array([y1_test, y2_test,y3_test])).flatten(), y_predict))
#에측 어느정도 하는지 예시
a=model.predict([[10],[107],[1035]])
print(a)

model.save('model_hose.h5')
