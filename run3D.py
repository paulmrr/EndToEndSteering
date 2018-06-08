from matplotlib import pyplot as plt
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, MaxPooling3D, Conv3D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
import sys
import os, shutil, fnmatch
import csv
import numpy as np
import random
#import cv2
import time

def build_model(shape):
    model = keras.Sequential()
    model.add(Conv3D(3, (5, 5, 5), activation='relu', input_shape=(shape)))
    model.add(MaxPooling3D((2, 2, 2)))
    model.add(MaxPooling3D((2, 2, 2)))
    model.add(Conv3D(64, (5, 5, 5), activation='relu'))
    model.add(Conv3D(64, (5, 5, 5), activation='relu'))
    model.add(Conv3D(8, (5, 5, 5), activation='relu'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')
    model.summary()
    return model

def generator3D(x, y, lookback, delay, min_index, max_index, shuffle=False, 
                batch_size=128, step=6):
    if max_index is None:
        max_index = len(x) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(min_index+lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)
        samples = np.zeros((len(rows), lookback//step, len(y)))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = x[indices]
            targets[j] = y[indicies]
    yield samples, targets

def train_model():
    print("loading...")
    data = np.load('dataset_0.25.npz', mmap_mode='r')
    X_train = data['train_data']/255
    y_train = data['train_labels']
    X_test = data['test_data']/255
    y_test = data['test_labels']

    lookback = 15
    step = 1
    delay = 0
    batch_size = 128

    train_gen = generator3D(X_train, y_train, lookback=lookback, delay=delay,
                          min_index=0, max_index=int(0.8*len(X_train)), 
                          shuffle=False, step=step, batch_size=batch_size)
    val_gen = generator3D(X_train, y_train, lookback=lookback, delay=delay, 
                        min_index=int(0.8*len(X_train))+1, 
                        max_index=None, step=step,
                        batch_size=batch_size)

    test_gen = generator3D(X_test, y_test, lookback=lookback, delay=delay,
                         min_index=0, max_index=None, step=step,
                         batch_size=batch_size)

    val_steps = len(X_train)-delay-1 - int(0.8*len(X_train) + 1) - lookback

    test_steps = (len(X_train) + len(X_test) - len(X_train) - lookback)
    
    print("starting...")

    model = build_model(X_train.shape[1:])
    model.fit_generator(train_gen, 
                        steps_per_epoch=int(0.8*len(X_train))//batch_size,
                        epochs=epochs, validation_data=val_gen,
                        validation_steps = len(X_train)//batch_size,
                        verbose=1)
    
    model.save('model3d.h5')
    print("saved model...")

    randint = np.random.randint(len(X_test), size=50)
    pred = model.predict(X_test[randint])
    actual = y_test[randint]
    Xs = X_test[randint]

    for i in range(len(actual)):
        print("Actual: {}, Predicted:".format(actual[i], pred[i]))

    print("ending...")

def main():
    train_model()

if __name__ == "__main__":
    main()
