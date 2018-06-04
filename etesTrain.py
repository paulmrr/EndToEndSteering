from keras import models, layers
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import sys
import os, fnmatch
import csv
import numpy as np
import random

# read csv files in and append to list, preparing for dictionary
steeringAngle = []
count = 1
while count < 11:
    with open(os.getcwd() + '/deeptesla/steer_' + '{0:0=2d}'.format(
             count)) as csvfile:
        read = csv.reader(csvfile)
        for row in read:
            steeringAngle.append(str(row)[1:-1])
    count += 1

# Create dictionary with namefile keys and steering angle values
imgExt = '*.png'
keys = []
for entry in sorted(os.listdir(os.getcwd() + '/deeptesla')):
    if fnmatch.fnmatch(entry, imgExt):
        keys.append(entry[0:3] + entry[8:12])

dataDict = dict(zip(keys, steeringAngle))

# Shuffler to toggle between Conv2D or 3D
def myShuffler(dataDict, Conv2or3D, sequenceSize=None):
    if Conv2or3D == 2:
        theKeys = list(dataDict.keys())
        random.shuffle(theKeys)
        data = theKeys
        trainDataKeys = data[0:int(0.6*len(data))]
        vldDataKeys = data[int(0.6*len(data)):int(0.8*len(data))]
        testDataKeys = data[int(0.8*len(data)):len(data)]
    if Conv2or3D == 3:
        # Separate data by sequence_size
        sequenceSize = sequenceSize
        currentSequence = 0
        sequenceDepth = []
        sequences = []
        for index, (k, v) in enumerate(sorted(dataDict.items())):
            if currentSequence <= (sequenceSize - 2):
                sequenceDepth.append(k)
            elif currentSequence > (sequenceSize - 2):
                sequences.append(sequenceDepth)
                print(sequences[-1])
                sequenceDepth = []
                sequenceDepth.append(k)
                currentSequence = -1
            currentSequence += 1
        random.shuffle(sequences)
        data = sequences
        trainDataKeys = data[0:int(0.6*len(data))]
        vldDataKeys = data[int(0.6*len(data)):int(0.8*len(data))]
        testDataKeys = data[int(0.8*len(data)):len(data)]

    return trainDataKeys, vldDataKeys, testDataKeys
        
# Create training data with 60% of data, 20% validation, 20% test, shuffling
trainDataKeys, vldDataKeys, testDataKeys = myShuffler(dataDict, 2)

trainDataKeys, vldDataKeys, testDataKeys = myShuffler(dataDict, 3, sequenceSize=8)
