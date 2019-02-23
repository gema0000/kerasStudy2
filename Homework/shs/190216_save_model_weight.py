
import numpy as np
import pandas as pd
import keras
from keras.layers import Dense, Activation, Concatenate, Input
from keras.models import Sequential, Model
from keras.layers.merge import concatenate
from dataset import x1_train, x2_train, x3_train, x1_test, x2_test, x3_test, y1_train, y2_train, y3_train, y1_test, y2_test, y3_test, merXtra, merYtra, merXtest, merYtest

# 모델 구성
# model 1
input1 = Input(shape=(1,))
dense1 = Dense(100, activation='relu')(input1)
# model 2
input2 = Input(shape=(1,))
dense2 = Dense(50, activation='relu')(input2)

# model 3
input3 = Input(shape=(1,))
dense3 = Dense(90, activation='relu')(input3)

# merge
# merge = Concatenate()([dense1, dense2])
merge1 = concatenate([dense1, dense2, dense3])

output_1 = Dense(100)(merge1)
output1 = Dense(1)(output_1)
output_2 = Dense(100)(merge1)
output2 = Dense(1)(output_2)
output_3 = Dense(100)(merge1)
output3 = Dense(1)(output_3)

# model = Model(inputs=[input1, input2, input3], outputs=[output1, output2, output3])
# model.compile(loss='mse', optimizer='adam', metrics=['mse'])
# # model.compile(loss='mse', optimizer='adam', metrics=['acc'])
# model.summary()

# x1_train = np.append(x1_train, np.ones(90)*x1train_avg)

model = Sequential()
model.add(Dense(100, input_dim = 12, activation='relu'))
# model.add(Dense(100000, input_shape=(2, ), activation='relu'))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(10))
# model.add(Flatten())
model.add(Dense(12))
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

# tensorboard 적용
tb_hist = keras.callbacks.TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)
# model.fit([x1_train, x2_train, x3_train], [y1_train, y2_train, y3_train] , epochs=100, batch_size=1,
#           validation_data=([x1_test, x2_test, x3_test], [y1_test, y2_test, y3_test]), callbacks=[tb_hist])
model.fit(merXtra, merYtra , epochs=100, batch_size=1,
          validation_data=(merXtest, merYtest), callbacks=[tb_hist])

# 모델, weight 저장
model.save('savedModelWeight.h5')
# a, b = model.evaluate([x1_test, x2_test],[y1_test, y2_test], batch_size=1)
# print(a, b)     
