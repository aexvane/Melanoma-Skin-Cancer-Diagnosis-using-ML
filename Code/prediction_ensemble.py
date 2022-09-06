from keras.models import load_model
import cv2
import numpy as np
import pandas as pd
import os

final_test = pd.read_csv("final_test.csv")

model = load_model('Ensemble.h5')

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['auc','accuracy'])

final_test = final_test.iloc[:,1:23]
print(np.shape(final_test))
prediction = model.predict_classes(final_test)
print(np.shape(prediction))
df = pd.DataFrame(prediction)
print(df.columns)

def my_conversion(x):
   if x == 0:
       return 1
   else :
       return 0

new = df.iloc[:,0].apply(lambda x : my_conversion(x))

new.to_csv('result_test_3.csv')