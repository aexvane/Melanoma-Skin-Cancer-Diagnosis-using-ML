from keras.models import load_model
import cv2
import numpy as np
import pandas as pd
import os

model = load_model('xception.h5')

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['auc','accuracy'])
image_size=(299, 299)
images=os.listdir(f"../prediction300/")
with open('prediction.csv', 'w') as f:

    for image in images:
        img = cv2.imread(f'../prediction300/{image}')
        img = cv2.resize(img, image_size)
        img = np.expand_dims(img, axis=0)

        classes = model.predict(img)

        print(classes)

        prediction_1 = classes[0][0]
        prediction_2 = classes[0][1]

        f.write(str(image)+","+str(prediction_1)+","+str(prediction_2)+"\n")

f.close()