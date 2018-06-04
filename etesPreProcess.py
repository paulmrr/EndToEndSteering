from keras import models, layers
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import sys
import os, shutil, fnmatch
import csv
import numpy as np
import random
import cv2

from imgs_to_npz import conv2d_npz_gen 

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
        keys.append(entry)

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

# Take shuffled keys and put images into lists ready for opencv
trainImages = []
vldImages = []
testImages = []
for entry in os.listdir(os.getcwd() + '/deeptesla'):
    for imageName in trainDataKeys:
        if imageName == entry:
            img = cv2.imread(os.getcwd() + '/deeptesla/' + str(imageName))
            #cv2.imshow('image', img)
            #cv2.waitKey(25)
            trainImages.append(img)
    for imageName in vldDataKeys:
        if imageName == entry:
            img = cv2.imread(os.getcwd() + '/deeptesla/' + str(imageName))
            vldImages.append(img)
    for imageName in testDataKeys:
        if imageName == entry:
            img = cv2.imread(os.getcwd() + '/deeptesla/' + str(imageName))
            testImages.append(img)

# Labels from the shuffled keys
trainLabels = []
for idx in trainDataKeys:
    trainLabels.append(float(dataDict[idx][1:-1]))

vldLabels = []
for idx in vldDataKeys:
    vldLabels.append(float(dataDict[idx][1:-1]))

testLabels = []
for idx in testDataKeys:
    testLabels.append(float(dataDict[idx][1:-1]))

conv2d_npz_gen("formatted_etes_05", trainImages, vldImages, testImages,.5)
conv2d_npz_gen("formatted_etes_025", trainImages, vldImages, testImages,.25)