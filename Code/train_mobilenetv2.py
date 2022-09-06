import os
import numpy as np
import random as rn
import string
import tensorflow as tf
import keras
import time
import matplotlib.pyplot as plt
from keras import initializers
from keras import regularizers
from keras import constraints
from keras import backend as K
from keras.activations import elu
from keras.optimizers import Adam
from keras.models import Sequential
from keras.engine import Layer, InputSpec
from keras.utils.generic_utils import get_custom_objects
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from keras.layers import Input, Dense, Conv2D, Flatten, GlobalAveragePooling2D, Dropout,MaxPooling2D,BatchNormalization,GlobalMaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.applications import MobileNetV2

IMG_SIZE = 256

epochs=3

seed=1234
rn.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ["PYTHONHASHSEED"]=str(seed)

base_model=MobileNetV2(input_shape=(IMG_SIZE,IMG_SIZE,3),include_top=False,weights="imagenet")

name = "mobilenetv2_yahya{}".format(int(time.time()))
tensorboard = TensorBoard( histogram_freq=1, log_dir= 'logs/{}'.format(name))

for layer in base_model.layers:
    layer.trainable=True

def build_model():
    model=Sequential()
    model.add(base_model) 
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.3))
    model.add(Dense(2,activation="sigmoid"))
    
    optimizer=keras.optimizers.Adam(learning_rate=0.00005,beta_1=0.9,beta_2=0.999,amsgrad=False)

    metrics=tf.keras.metrics.AUC(name="auc")

    model.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=[metrics,'accuracy'])
    print(model.summary())
    return model

model=build_model()

def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.savefig(f'./mobilenetv2.png')
    plt.show()

src_path_train = "../resized/"

train_datagen = ImageDataGenerator(
        validation_split=0.20)

batch_size = 32
train_generator = train_datagen.flow_from_directory(
    directory=src_path_train,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    subset='training',
    shuffle=True,
    seed=seed
)
valid_generator = train_datagen.flow_from_directory(
    directory=src_path_train,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    subset='validation',
    shuffle=True,
    seed=seed
)

checkpoint = ModelCheckpoint(f'./mobilenetv2.h5',verbose=1, save_best_only=False,
    save_weights_only=False, mode='auto', save_freq='epoch',
    options=None)

# early_stopping_monitor = EarlyStopping(
#     monitor='val_loss',
#     min_delta=0,
#     patience=0,
#     verbose=0,
#     mode='auto',
#     baseline=None,
#     restore_best_weights=True
# )

hist=model.fit(
    train_generator,
    epochs=epochs,
    shuffle=True,
    validation_data=valid_generator,
    steps_per_epoch = train_generator.n//train_generator.batch_size,
    validation_steps = valid_generator.n//valid_generator.batch_size,
    verbose=1,

    callbacks=[checkpoint, tensorboard]
)
plot_hist(hist)