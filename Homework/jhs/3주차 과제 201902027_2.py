from keras.models import load_model

model2 = load_model('model_hose.h5')

b=model2.predict([[10],[107],[1035]])
