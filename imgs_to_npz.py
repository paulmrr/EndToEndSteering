# AutonomousVehicleGuidanceUsingDLNN
# Journery McDowell
# Paul Rothhammer-Ruiz
# Chris Veranese
# 
# imgs_to_npz.py
#
# Description: This file test out saving image and steering files to npz to later load them 
# 
# Date: June 3, 2018

import numpy as np
import cv2
import os.path

full_path = os.getcwd() + '/deeptesla/'
npz_path = os.getcwd()+"/" 


def progress(iteration,index,size):
    if (iteration == 0):
        data_class = "train"
    elif (iteration == 1):
        data_class = "val"
    elif (iteration == 2):
        data_class = "test"
    p = float(index/size)
    if (p > 0  and p < .05 ):
        print('\riteration={}  |~~~~~~~~~~~~~~~~~~~~|'.format(data_class),end="") 
    if (p > .05  and p < .1 ):
        print('\riteration={}  |>~~~~~~~~~~~~~~~~~~~|'.format(data_class),end="") 
    if (p > .1  and p < .15 ):
        print('\riteration={}  |=>~~~~~~~~~~~~~~~~~~|'.format(data_class),end="") 
    if (p > .15  and p < .2 ):
        print('\riteration={}  |==>~~~~~~~~~~~~~~~~~|'.format(data_class),end="") 
    if (p > .2  and p < .25 ):
        print('\riteration={}  |===>~~~~~~~~~~~~~~~~|'.format(data_class),end="") 
    if (p > .25  and p < .3 ):
        print('\riteration={}  |====>~~~~~~~~~~~~~~~|'.format(data_class),end="") 
    if (p > .3  and p < .35 ):
        print('\riteration={}  |=====>~~~~~~~~~~~~~~|'.format(data_class),end="") 
    if (p > .35  and p < .4 ):
        print('\riteration={}  |======>~~~~~~~~~~~~~|'.format(data_class),end="") 
    if (p > .4  and p < .45 ):
        print('\riteration={}  |=======>~~~~~~~~~~~~|'.format(data_class),end="") 
    if (p > .45  and p < 5 ):
        print('\riteration={}  |========>~~~~~~~~~~~|'.format(data_class),end="") 
    if (p > .5  and p < .55 ):
        print('\riteration={}  |=========>~~~~~~~~~~|'.format(data_class),end="") 
    if (p > .55  and p < .6 ):
        print('\riteration={}  |==========>~~~~~~~~~|'.format(data_class),end="") 
    if (p > .6  and p < .65 ):
        print('\riteration={}  |===========>~~~~~~~~|'.format(data_class),end="") 
    if (p > .65  and p < .7 ):
        print('\riteration={}  |============>~~~~~~~|'.format(data_class),end="") 
    if (p > .7  and p < .75 ):
        print('\riteration={}  |=============>~~~~~~|'.format(data_class),end="") 
    if (p > .75  and p < .7 ):
        print('\riteration={}  |==============>~~~~~|'.format(data_class),end="") 
    if (p > .8  and p < .85 ):
        print('\riteration={}  |===============>~~~~|'.format(data_class),end="") 
    if (p > .85  and p < .9 ):
        print('\riteration={}  |================>~~~|'.format(data_class),end="") 
    if (p > .9  and p < .95 ):
        print('\riteration={}  |=================>~~|'.format(data_class),end="") 
    if (p > .95  and p < 1 ):
        print('\riteration={}  |==================>~|'.format(data_class),end="") 



def conv2d_npz_gen(filename, train_data,val_data,test_data, resize):
    dataset_size = []
    dataset_size.append(len(train_data))
    dataset_size.append(len(val_data))
    dataset_size.append(len(test_data))

    train_array = []
    val_array = []
    test_array = []
    for i in range(3):
        #shuffled_images = [x+1 for x in range(dataset_size[i])] 
        #random.shuffle(shuffled_images) 
        for j in range(dataset_size[i]):
            progress(i,j,dataset_size[i])
            if (i == 0):
                orig_img = train_data[j] 
                # crop image by removing top 300 pixles
                crop_img = orig_img[300:720, 0:1280]
                # resize image by scaling down by 2 
                resize_img = cv2.resize(crop_img, (0,0), fx=resize, fy=resize) 
                train_array.append(resize_img)    
            elif (i == 1):
                orig_img = val_data[j] 
                # crop image by removing top 300 pixles
                crop_img = orig_img[300:720, 0:1280]
                # resize image by scaling down by 2 
                resize_img = cv2.resize(crop_img, (0,0), fx=resize, fy=resize) 
                val_array.append(resize_img)    
            elif (i == 2):
                orig_img = test_data[j] 
                # crop image by removing top 300 pixles
                crop_img = orig_img[300:720, 0:1280]
                # resize image by scaling down by 2 
                resize_img = cv2.resize(crop_img, (0,0), fx=resize, fy=resize) 
                test_array.append(resize_img)    
            
            
            
    print()
    train_array = np.array(train_array)
    val_array = np.array(val_array)
    test_array = np.array(test_array)
    print("Loading images into npz file...")

    np.savez_compressed(npz_path+filename, train=train_array, val=val_array, test=test_array)
    print()
 
