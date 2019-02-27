from keras.models import Sequential,Model
from keras.layers import LSTM, Dense,Input
from keras import optimizers
import numpy as np
import pandas as pd
import keras.backend as K
from keras.models import load_model
from keras.layers.merge import concatenate
from keras.models import load_model

#데이터 생성
train1=np.array(range(1,111))
train1=pd.DataFrame(train1)

train2=np.array(range(1001,1111))
train2=pd.DataFrame(train2)

for s in range(1, 11):
    train1['shift_{}'.format(s)] = train1[0].shift(-s)
    train2['shift_{}'.format(s)] = train2[0].shift(-s)
    
train1=train1.dropna()
train2=train2.dropna()

x1=train1.iloc[:,0:10]
y1=train1.iloc[:,-1]
x2=train2.iloc[:,0:10]
y2=train2.iloc[:,-1]

input1=Input(shape=(10,1))
lstm1=LSTM(30)(input1)
dense1=Dense(50,activation='relu',kernel_initializer='he_uniform')(lstm1)
dense11=Dense(1,activation='relu',kernel_initializer='he_uniform')(dense1)

input2=Input(shape=(10,1))
lstm2=LSTM(30)(input2)
dense2=Dense(50,activation='relu',kernel_initializer='he_uniform')(lstm2)
dense22=Dense(1,activation='relu',kernel_initializer='he_uniform')(dense2)
merge1=concatenate([dense11, dense22])
model = Model(inputs=[input1, input2], outputs=[merge1])

x1=np.array(x1)

x1=x1.reshape(-1,10,1)

x2=np.array(x2)

x2=x2.reshape(-1,10,1)

y=pd.concat([y1,y2],axis=1)

y=np.array(y)

model.compile(loss='mse', optimizer='adamax', metrics=['mse'])

model.fit([x1,x2],[y],epochs=500,batch_size=1)

model.predict([x1[1].reshape(1,10,1),x2[1].reshape(1,10,1)])

model.save('model_hose.h5')
