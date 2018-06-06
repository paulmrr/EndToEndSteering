# AutonomousVehicleGuidanceUsingDLNN
# Paul Rothhammer-Ruiz
# Chris Veranese
# Journey McDowell
# 
# crop_resize.py
#
# Description: This file test out saving image and steering files to npz to later load them 
# 
# Date: June 3, 2018

import numpy as np
import cv2
import os.path
import re
import shutil
import os
from os import listdir
import random
import sys


RATIO_TRAIN_TEST = .8
IMG_DIR = "deeptesla"
DATA_PATH = "/Users/Paul/Documents/github/E2ES_data"
RESIZE = .25
DIRECTORY = "e2es_images_"+str(RESIZE)
NPZ_NAME = "/dataset_"+str(RESIZE)

def progress(iteration,index,size):
    if (iteration == 0):
        data_class = "train"
    elif (iteration == 1):
        data_class = "val"
    elif (iteration == 2):
        data_class = "test"
    else:
        data_class = iteration
    p = float(index/size)
    if (p > 0  and p < .05 ):
        print('\r{}  |~~~~~~~~~~~~~~~~~~~~|'.format(data_class),end="") 
    if (p > .05  and p < .1 ):
        print('\r{}  |>~~~~~~~~~~~~~~~~~~~|'.format(data_class),end="") 
    if (p > .1  and p < .15 ):
        print('\r{}  |=>~~~~~~~~~~~~~~~~~~|'.format(data_class),end="") 
    if (p > .15  and p < .2 ):
        print('\r{}  |==>~~~~~~~~~~~~~~~~~|'.format(data_class),end="") 
    if (p > .2  and p < .25 ):
        print('\r{}  |===>~~~~~~~~~~~~~~~~|'.format(data_class),end="") 
    if (p > .25  and p < .3 ):
        print('\r{}  |====>~~~~~~~~~~~~~~~|'.format(data_class),end="") 
    if (p > .3  and p < .35 ):
        print('\r{}  |=====>~~~~~~~~~~~~~~|'.format(data_class),end="") 
    if (p > .35  and p < .4 ):
        print('\r{}  |======>~~~~~~~~~~~~~|'.format(data_class),end="") 
    if (p > .4  and p < .45 ):
        print('\r{}  |=======>~~~~~~~~~~~~|'.format(data_class),end="") 
    if (p > .45  and p < 5 ):
        print('\r{}  |========>~~~~~~~~~~~|'.format(data_class),end="") 
    if (p > .5  and p < .55 ):
        print('\r{}  |=========>~~~~~~~~~~|'.format(data_class),end="") 
    if (p > .55  and p < .6 ):
        print('\r{}  |==========>~~~~~~~~~|'.format(data_class),end="") 
    if (p > .6  and p < .65 ):
        print('\r{}  |===========>~~~~~~~~|'.format(data_class),end="") 
    if (p > .65  and p < .7 ):
        print('\r{}  |============>~~~~~~~|'.format(data_class),end="") 
    if (p > .7  and p < .75 ):
        print('\r{}  |=============>~~~~~~|'.format(data_class),end="") 
    if (p > .75  and p < .7 ):
        print('\r{}  |==============>~~~~~|'.format(data_class),end="") 
    if (p > .8  and p < .85 ):
        print('\r{}  |===============>~~~~|'.format(data_class),end="") 
    if (p > .85  and p < .9 ):
        print('\r{}  |================>~~~|'.format(data_class),end="") 
    if (p > .9  and p < .95 ):
        print('\r{}  |=================>~~|'.format(data_class),end="") 
    if (p > .95  and p < 1 ):
        print('\r{}  |===================>|'.format(data_class),end="") 


def get_img_list(img_path):
    img_list = [f for f in listdir(img_path)]
    total_num_files = len(img_list)
    return img_list


def crop_and_resize(img_path,dir_path,resize):
    img_list = [f for f in listdir(img_path)]
    total_num_files = len(img_list)
    

    iteration = 0 
    img_num = 0
    images_processed = 0
    print("Starting Cropping and Resizing")
    for img_name in img_list:
        if (images_processed % 100 == 0):
            progress("Crop and Resizing",images_processed,total_num_files)
        images_processed+=1

        orig_img = cv2.imread(img_path+"/"+img_name)  
        # crop image by removing top 300 pixles
        crop_img = orig_img[300:720, 0:1280]
        # resize image by scaling down by 2 
        resize_img = cv2.resize(crop_img, (0,0), fx=resize, fy=resize) 
        cv2.imwrite(dir_path+"/"+img_name,resize_img)

    
    print("Finished Cropping and Resizing\n")
    return img_list



def save_to_npz(img_path,img_list):
    img_path = img_path
    img_list = img_list
    total_num_img = len(img_list)

    img_array = []
    label_array = []

    print("\nSaving image names and steer_values")
    img_dict = {}
    img_count = 0
    images_processed = 0
    for img_name in img_list:
        if (images_processed % 100 == 0):
            progress("Dictionary of images + steering",images_processed,total_num_img)
        images_processed+=1

        
        p = re.compile('[0-9][0-9]_image[0-9][0-9][0-9][0-9]_(.*).png') 
        steer_value = p.findall(img_name)
        if (len(steer_value) == 0):
            continue
        steer_value = steer_value[0]
        img_dict[img_name] = steer_value


    print("\nLoading Images to array")
    images_processed = 0
    for key, value in sorted(img_dict.items(), key=lambda img_dict: random.random()):
        if (images_processed % 50 == 0):
            progress("Generating Numpy Arrays of Images",images_processed,total_num_img)
        images_processed+=1

        img = cv2.imread(img_path+"/"+key) 
        img_array.append(img)
        label_array.append(value) 
        
    label_array = np.array(label_array)
    img_array = np.array(img_array)

    num_train = int(total_num_img*.9)

    print("\nSaving images into NPZ format")
    data = np.savez_compressed(os.getcwd()+NPZ_NAME, train_data=img_array[0:num_train],train_labels=label_array[0:num_train],test_data=img_array[num_train:],test_labels=label_array[num_train:])


def main():
    full_path = DATA_PATH 
    old_img_path = full_path+"/"+IMG_DIR
    new_img_path = full_path+"/"+DIRECTORY
    if (len(sys.argv) < 2):
        print("Missing CLA! Add CLA of < CROP > in order to crop and resize images OR Add CLA of < NOCROP > to save to .npz without cropping")
        return 
    if (sys.argv[1] == "CROP"):
        # Delete directory if exists and create new directory to save images
        if not os.path.exists(full_path+"/"+DIRECTORY):
            os.makedirs(full_path+"/"+DIRECTORY)
            print("\nNew directory made with path {}\n".format(full_path+"/"+DIRECTORY))
        else:
            
            shutil.rmtree(full_path+"/"+DIRECTORY)
            print("\nDirectory with path {} deleted".format(full_path+"/"+DIRECTORY))
            os.makedirs(full_path+"/"+DIRECTORY)
            print("New directory made with path {}\n".format(full_path+"/"+DIRECTORY))

        resize1 = RESIZE
        img_list = crop_and_resize(old_img_path,new_img_path,resize1)
        save_to_npz(new_img_path,img_list)
    elif (sys.argv[1] == "NOCROP"):
        if not os.path.exists(new_img_path):
            img_list = get_img_list(new_img_path)
            save_to_npz(new_img_path,img_list)
        else: 
            print(str(new_img_path)+" DOES NOT EXIST!!!")
    else:
        print("WRONG CLA! Add CLA of CROP in order to crop and resize images OR Add CLA of NOCROP to save to npz without cropping")
        return
    
    print("Done!")



if __name__=="__main__":
    main()
