import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, MaxPooling3D, Conv3D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import sys
import os
import numpy as np
import numpy.random as rnd
#import cv2
import time

startTime = time.time()

def build_model(shape):
    model = keras.Sequential()
    model.add(Conv3D(16, (3, 3, 3), activation='relu', input_shape=(shape)))
    #model.add(Conv3D(32, (5, 5, 5), activation='relu'))
    model.add(MaxPooling3D((2, 2, 2)))
    #model.add(Conv3D(64, (3, 3, 3), activation='relu'))
    #model.add(Conv3D(64, (3, 3, 3), activation='relu'))
    #model.add(MaxPooling3D((2, 2, 2)))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(4))

    model.compile(optimizer='adam', loss='mse')
    model.summary()
    return model

def generator3D(x, y, sequence_len, batch_size):
    seq_list = []
    batch_list = []
    train_size = len(x)
    random_seq = [i for i in range(int(len(x)/sequence_len))]
    rnd.shuffle(random_seq)
    count = 0
    while True:
        batch_list = []
        labels = []
        for i in range(batch_size):
            if count >= len(random_seq):
                count = 0
                rnd.shuffle(random_seq)
            batch_list.append(x[random_seq[count]*sequence_len:                                                           (random_seq[count]+1)*sequence_len])
            labels.append(y[random_seq[count]*sequence_len:
                          (random_seq[count]+1)*sequence_len])
            count += 1
        yield np.array(batch_list), np.array(labels)

def train_model():
    print("loading...")
    print()

    data = np.load('dataset_0.25.npz', mmap_mode='r')
    X_train = data['train_data'] / 255
    y_train = data['train_labels']
    X_test = data['test_data'] / 255
    y_test = data['test_labels']
    
    loadTime = time.time() - startTime
    print('Load Time: ', loadTime)
    print()

    sequence_len = 4
    batch_size = 32
    epochs = 8

    train_gen = generator3D(X_train[0:21609], y_train[0:21609], sequence_len, batch_size)

    val_gen = generator3D(X_train[21609:], y_train[21609:], sequence_len, batch_size)

    test_gen = generator3D(X_test, y_test, sequence_len, batch_size)


    print('input shape: ', next(train_gen)[0][0].shape)
    print()
    
    model = build_model(next(train_gen)[0][0].shape)
    
    print("starting...")
    print()
    
    hist = model.fit_generator(train_gen, 
                               steps_per_epoch=len(X_train[0:21609])//batch_size,
                               epochs=epochs, validation_data=val_gen,
                               validation_steps = len(X_train[21609:])//batch_size,
                               verbose=1)
    
    model.save("model3d_4.h5")
    print("saved model...")
    print()
    
    trainTime = time.time() - startTime - loadTime
    
    print('train time: ', trainTime)
    print()

    plotFlag = True
    if plotFlag:
        loss = hist.history['loss']
        val_loss = hist.history['val_loss']
        epoch_range = range(1, len(loss) + 1)
        fig = plt.figure()
        plt.plot(epoch_range, loss, 'bo', label='Training loss')
        plt.plot(epoch_range, val_loss, 'b', label='Validation loss')
        plt.title('Loss v. epoch')
        plt.legend()
        fig.savefig('Conv3D_seq_{}'.format(sequence_len))
        #plt.show()

    print("ending...")

def main():
    train_model()

if __name__ == "__main__":
    main()
