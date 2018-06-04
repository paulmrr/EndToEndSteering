import numpy as np
import cv2

npz_file = np.load('etes_data_formated_05.npz')
print(npz_file.files)

#for i in range(10):
#    pic = (npz_file['data01'][i])
#    print( (npz_file['data01'][i]).shape)
#    cv2.imshow('image',pic)
#    cv2.waitKey(5)
pic = (npz_file['val'][40])
cv2.imshow('image',pic)
cv2.waitKey(1000)
print("nextImage)")


pic = (npz_file['test'][23])
cv2.imshow('image',pic)
cv2.waitKey(1000)

pic = (npz_file['train'][41])
cv2.imshow('image',pic)
cv2.waitKey(1000)
