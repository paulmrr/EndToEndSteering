import numpy as np
import cv2

# with np.load('dataset_.5.npz', mmap_mode='r') as data:
#     train = data['train_data'][0]
#     print(' ')

data = np.load('data/train_data.npy', mmap_mode='r')

print(data.shape)

pic = (data[40])
print(pic)
cv2.imshow('image',pic)
cv2.waitKey(0)
print("nextImage)")

# pic = (npz_file['train_labels'][40])
# cv2.imshow('image',pic)
# cv2.waitKey(1000)
# print("nextImage)")
