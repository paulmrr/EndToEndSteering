import cv2
import math
from keras.models import load_model
import numpy as np
from vis.visualization import visualize_cam, overlay

# titles = ['right steering', 'left steering', 'maintain steering']
model = load_model("model.h5")
imgs = np.load("demo_images.npy")
# input needs to be converted to 0-1 instead of 0-255
show_imgs = imgs/255
labs = np.load("demo_labels.npy")

predictions = model.predict(show_imgs)

for i in range(len(imgs)):
    img = imgs[i]
    lab = str(labs[i])
    pred = predictions[i]

    # actual saliency visualization
    # heatmapR = visualize_cam(model, layer_idx=-1, filter_indices=0, 
    #         seed_input=show_imgs[i], grad_modifier=None)
    # overlay heatmaps onto the seed image
    # overlay1 = overlay(img, heatmapL, alpha=.7)

    # put images together
    # fin = np.hstack((overlay1, overlay2, overlay3))
    start_point = (len(img[0])//2, len(img))
    print(pred[0])
    end_point = (
        int(start_point[0] + 100*math.cos(pred[0]*math.pi/180 - math.pi/2)),
        int(start_point[1] + 100*math.sin(pred[0]*math.pi/180 - math.pi/2))
    )

    cv2.line(img, start_point, end_point, (255, 0, 0), 3) 

    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
