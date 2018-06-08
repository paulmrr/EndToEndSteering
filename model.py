import keras
from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, Dropout
import numpy as np

img_size = (210, 640, 3)

def build_model():
    
    model = keras.Sequential()
    model.add(Conv2D(32, (5, 5), activation='relu',
              input_shape=(img_size)))
    model.add(Conv2D(32, (5, 5), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='sigmoid'))
    model.add(Dropout(.5))

    model.add(Dense(1))

    model.compile(
        optimizer='adam',
        loss='mse',
    )
    print(model.summary())

    return model

def train_model(model):
    # X_train = np.load('data/train_data.npy', mmap_mode='r')
    # y_train = np.load('data/train_labels.npy', mmap_mode='r')
    # X_test = np.load('data/test_data.npy', mmap_mode='r')
    # y_test = np.load('data/test_labels.npy', mmap_mode='r')

    data = np.load('data.npz')

    X_train = data['train_data']
    y_train = data['train_labels']
    X_test = data['test_data']
    y_test = data['test_labels']

    print(X_train.shape)
    print(y_train.shape)

    print("starting...")
    # model.fit(X_train, y_train, batch_size=32, epochs=5,
    #           validation_data=(X_test, y_test))
    model.fit(X_train, y_train, batch_size=32, epochs=5)
    print("ending...")

def main():
    model = build_model()
    train_model(model)


if __name__ == "__main__":
    main()
