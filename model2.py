import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, Dropout
import numpy as np

def build_model(shape):
    
    model = keras.Sequential()
    model.add(Conv2D(32, (5, 5), activation='relu',
              input_shape=(shape)))
    model.add(Conv2D(32, (5, 5), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(.5))

    model.add(Dense(1))

    model.compile(
        optimizer='adam',
        loss='mse',
    )
    print(model.summary())

    return model

def train_model():
    data = np.load('dataset_0.25.npz', mmap_mode='r')
    X_train = data['train_data']/255
    y_train = data['train_labels']
    X_test = data['test_data']/255
    y_test = data['test_labels']

    # data = np.load('data.npz', mmap_mode='r')

    # X_train = data['train_data'][:500]
    # y_train = data['train_labels'][:500]
    # X_test = data['test_data']
    # y_test = data['test_labels']


    # print(X_train.shape)
    # print(y_train.shape)
    # model = build_model(X_train.shape[1:])

    print("starting...")
    model.fit(X_train, y_train, batch_size=32, epochs=2,
        validation_data=(X_test, y_test))

    # model = load_model("model.h5")

    model.save("model.h5")

    randint = np.random.randint(len(X_test), size=50)
    pred = model.predict(X_test[randint])
    actual = y_test[randint]
    Xs = X_test[randint]

    for i in range(len(actual)):
        print("Actual: {}, Predicted: {}".format(actual[i], pred[i]))
        # print("X_train: {}".format(Xs[i]))

    print("ending...")

def main():
    train_model()


if __name__ == "__main__":
    main()
