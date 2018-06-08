import cv2
import numpy as np
from matplotlib import pyplot as plt

data = np.load("dataset_0.25.npz")
index = 500
# imgs = data['test_data']
# labs = data['test_labels']

imgs = data['test_data']
labs = data['test_labels']
img = np.load("img1.npy")
indexs = np.random.randint(len(imgs), size=10)
np.save("demo_images", imgs[indexs])
np.save("demo_labels", labs[indexs])

# cv2.imshow('image', img)
# cv2.waitKey(0)
#cv2.destroyAllWindows()



