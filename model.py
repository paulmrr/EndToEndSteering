import keras
from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, Conv2D
import numpy as np

img_size = (260, 640, 3)

def build_model():
    
    model = keras.Sequential()
    model.add(Conv2D(32, (5, 5), activation='relu',
              input_shape=(img_size)))
    model.add(Conv2D(32, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(Dense(128, activation='sigmoid'))
    model.add(Dropout(.5))

    model.add(Dense(1))

    model.compile(
        optimizer='adam',
        loss='mse',
    )

    return model

def train_model(model):
    X_train, y_train, X_test, y_test = load_tesla_data()

    model.fit(X_train, y_train, batch_size=32, epochs=5,
              validation_data=(X_test, y_test))

