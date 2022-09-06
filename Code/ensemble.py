import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import sklearn
from keras import losses
from keras import optimizers
from keras import metrics
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

file = pd.read_csv('final.csv')

X   = file.iloc[:,1:23]

Y   = file.iloc[:,23:25]

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.33, random_state=42)

model = keras.Sequential(
    [
        layers.Dense(32, activation=None, name="layer1",input_shape=(22,)),
        layers.Dense(64, activation=None, name="layer2"),
        layers.Dense(128, activation="relu", name="layer3"),
        layers.Dense(256, activation="relu", name="layer4"),
        layers.Dense(512, activation="relu", name="layer5"),
        layers.Dense(2, activation="sigmoid", name="layer6"),
    ]
)
model.compile(loss = 'binary_crossentropy',  
optimizer=keras.optimizers.Adam(learning_rate=0.00005,beta_1=0.9,beta_2=0.999,amsgrad=False), 
metrics = ["accuracy"])

def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.savefig(f'./Ensemble_acc.png')
    plt.show()

def plot_hist_1(hist):
    plt.plot(hist.history["loss"])
    plt.plot(hist.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.savefig(f'./Ensemble_loss.png')
    plt.show()

checkpoint = ModelCheckpoint(f'./Ensemble_new.h5',verbose=1, save_best_only=False,
    save_weights_only=False, mode='auto', save_freq='epoch',
    options=None)

hist = model.fit( X_train, y_train,
              batch_size=64, epochs=32,
              verbose=1,
              validation_data=(X_test, y_test),
              callbacks=[checkpoint]
)

score = model.evaluate(X_test, y_test, verbose=1)

print(score)

plot_hist(hist)

plot_hist_1(hist)