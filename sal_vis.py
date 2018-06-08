import cv2
from keras.models import load_model
import numpy as np
from vis.visualization import visualize_cam, overlay

# titles = ['right steering', 'left steering', 'maintain steering']
modifiers = [None, 'negate', 'small_values']

model = load_model("model.h5")
imgs = np.load("demo_images.npy")
# input needs to be converted to 0-1 instead of 0-255
show_imgs = imgs/255
labs = np.load("demo_labels.npy")

predictions = model.predict(show_imgs)

for i in range(len(imgs)):
    img = imgs[i]
    lab = str(labs[i])
    pred = str(predictions[i])

    # actual saliency visualization
    heatmapR = visualize_cam(model, layer_idx=-1, filter_indices=0, 
            seed_input=show_imgs[i], grad_modifier=None)

    heatmapL = visualize_cam(model, layer_idx=-1, filter_indices=0, 
            seed_input=show_imgs[i], grad_modifier='negate')

    heatmapM = visualize_cam(model, layer_idx=-1, filter_indices=0, 
            seed_input=show_imgs[i], grad_modifier='small_values')

    # overlay heatmaps onto the seed image
    overlay1 = overlay(img, heatmapL, alpha=.7)
    overlay2 = overlay(img, heatmapM, alpha=.7)
    overlay3 = overlay(img, heatmapR, alpha=.7)

    # put images together
    fin = np.hstack((overlay1, overlay2, overlay3))

    cv2.imshow(str(pred + ', ' + lab), fin)
    cv2.imwrite("saliency.png", fin)
    # cv2.imshow("img", fin)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
