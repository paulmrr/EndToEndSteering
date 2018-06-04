# AutonomousVehicleGuidanceUsingDLNN
# Journery McDowell
# Paul Rothhammer-Ruiz
# Chris Veranese
# 
# Description: This file test out saving image and steering files to npz to later load them 
# 
# Date: June 3, 2018 


#Example data extraction from keras dataset

import numpy as np
import cv2
import os.path
import random

def progress(iteration,index,size):
    p = float(index/size)
    if (p > 0  and p < .1 ):
        print('\riteration={}  |~~~~~~~~~~|'.format(iteration+1),end="") 
    if (p > .1  and p < .2 ):
        print('\riteration={}  |>~~~~~~~~~|'.format(iteration+1),end="") 
    if (p > .2  and p < .3 ):
        print('\riteration={}  |=>~~~~~~~~|'.format(iteration+1),end="") 
    if (p > .3  and p < .4 ):
        print('\riteration={}  |==>~~~~~~~|'.format(iteration+1),end="") 
    if (p > .4  and p < .5 ):
        print('\riteration={}  |===>~~~~~~|'.format(iteration+1),end="") 
    if (p > .5  and p < .6 ):
        print('\riteration={}  |====>~~~~~|'.format(iteration+1),end="") 
    if (p > .6  and p < .7 ):
        print('\riteration={}  |=====>~~~~|'.format(iteration+1),end="") 
    if (p > .7  and p < .8 ):
        print('\riteration={}  |======>~~~|'.format(iteration+1),end="") 
    if (p > .8  and p < .9 ):
        print('\riteration={}  |=======>~~|'.format(iteration+1),end="") 
    if (p > .9  and p < 1 ):
        print('\riteration={}  |========>~|'.format(iteration+1),end="") 
    




NUM_DATASETS = 10
#full_path = "/Users/Paul/Documents/github/EndToEndSteering/deeptesla/dataset"

full_path = "/home/journey/Documents/deeptesla-short/"
#path = "/Users/Paul/Documents/github/EndToEndSteering/"
#train_dir = "etes_data/train"
#test_dir = "etes_data/train"
#val_dir = "etes_data/train"

path_list = []

# range is from 1-10 instead of 0-9
for count in range(NUM_DATASETS):
    path_list.append("{0:02d}".format(count + 1))

print(path_list)
steer_dict1 = {}
steer_dict2 = {}
steer_dict3= {}
steer_dict4= {}
steer_dict5= {}
steer_dict6= {}
steer_dict7= {}
steer_dict8= {}
steer_dict9= {}
steer_dict10= {}
dataset_size = []

for dataset_num in range(10):
    with open(full_path+"steer"+"_"+"{0:02d}".format(dataset_num+1),"r") as steer_file:
        num_files = 0
        for line_num, steer_val in enumerate(steer_file):
            num_files += 1
            if (line_num+1 == 1):
                steer_dict1[str(line_num)] = float(steer_val.strip('\n')) 
            elif (line_num+1 == 2):
                steer_dict2[str(line_num)] = float(steer_val.strip('\n')) 
            elif (line_num+1 == 3):
                steer_dict3[str(line_num)] = float(steer_val.strip('\n')) 
            elif (line_num+1 == 4):
                steer_dict4[str(line_num)] = float(steer_val.strip('\n')) 
            elif (line_num+1 == 5):
                steer_dict5[str(line_num)] = float(steer_val.strip('\n')) 
            elif (line_num+1 == 6):
                steer_dict6[str(line_num)] = float(steer_val.strip('\n')) 
            elif (line_num+1 == 7):
                steer_dict7[str(line_num)] = float(steer_val.strip('\n')) 
            elif (line_num+1 == 8):
                steer_dict8[str(line_num)] = float(steer_val.strip('\n')) 
            elif (line_num+1 == 9):
                steer_dict9[str(line_num)] = float(steer_val.strip('\n')) 
            elif (line_num+1 == 10):
                steer_dict10[str(line_num)] = float(steer_val.strip('\n')) 
    dataset_size.append(num_files)

images_array1 = []
images_array2 = []
images_array3 = []
images_array4 = []
images_array5 = []
images_array6 = []
images_array7 = []
images_array8 = []
images_array9 = []
images_array10 = []
for i in range(NUM_DATASETS):
    #shuffled_images = [x+1 for x in range(dataset_size[i])] 
    #random.shuffle(shuffled_images) 
    for j in range(dataset_size[i]+1):
        progress(i,j,dataset_size[i])
        img_name = "image"+str("{0:04d}".format(j+1)) # Assumes numbering of images starts at 0 
        img_path = full_path+"{0:02d}".format(i+1)+"_"+img_name+".png"
        if (not os.path.isfile(img_path)):
            break 
        orig_img = cv2.imread(img_path) 
        # crop image by removing top 300 pixles
        crop_img = orig_img[300:720, 0:1280]
        # resize image by scaling down by 2 
        resz_img = cv2.resize(crop_img, (0,0), fx=0.5, fy=0.5) 
        orig_img = resz_img
        # if file does not exist continue to next dataset
        if (i+1 == 1):
            images_array1.append(orig_img)    
        elif (i+1 == 2):
            images_array2.append(orig_img)    
        elif (i+1 == 3):
            images_array3.append(orig_img)    
        elif (i+1 == 4):
            images_array4.append(orig_img)    
        elif (i+1 == 5):
            images_array5.append(orig_img)    
        elif (i+1 == 6):
            images_array6.append(orig_img)    
        elif (i+1 == 7):
            images_array7.append(orig_img)    
        elif (i+1 == 8):
            images_array8.append(orig_img)    
        elif (i+1 == 9):
            images_array9.append(orig_img)    
        elif (i+1 == 10):
            images_array10.append(orig_img)    
images_array1 = np.array(images_array1)
print()
images_array2 = np.array(images_array3)
images_array3 = np.array(images_array3)
images_array4 = np.array(images_array4)
images_array5 = np.array(images_array5)
images_array6 = np.array(images_array6)
images_array7 = np.array(images_array7)
images_array8 = np.array(images_array8)
images_array9 = np.array(images_array9)
images_array10 = np.array(images_array10)
print("Loading images into npz file...")

np.savez_compressed("/home/journey/Documents/dataset_.5resize", data01=images_array1, data02=images_array2, data03=images_array3, data04=images_array4, data05=images_array5, data06=images_array6, data07=images_array7, data08=images_array8, data09=images_array9, data10=images_array10)
   
#data = np.savez_compressed("/Users/Paul/Desktop/dataset_.5resize", train=train_array, val=val_array, test=test_array)

#def load_data(path='mnist.npz'):
#    """"Loads the MNIST dataset.
#
#    # Arguments
#        path: path where to cache the dataset locally
#            (relative to ~/.keras/datasets).
#
#    # Returns
#        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
#    """"
#    path = get_file(path,
#                    origin='https://s3.amazonaws.com/img-datasets/mnist.npz',
#                    file_hash='8a61469f7ea1b51cbae51d4f78837e45')
#    f = np.load(path)
#    x_train, y_train = f['x_train'], f['y_train']
#    x_test, y_test = f['x_test'], f['y_test']
#    f.close()
#    return (x_train, y_train), (x_test, y_test)
#"""


