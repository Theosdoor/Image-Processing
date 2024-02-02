import sys
import os

import cv2
import numpy as np

# path_to_images = sys.argv[1] # path given as command line argument
# check if valid path? like in ai search
path_to_images = 'image_processing_files/xray_images' # TODO REMOVE LATER to allow arg

# create results directory if doesnt already exist
if not os.path.isdir('Results'):
    os.mkdir('Results')

# dictionary matching image name to image matrix
images = {}

# load images
count = 0
for img in os.listdir(path_to_images):
    if count > 0: break # TODO only load first image for now
    if not img.endswith('.jpg'):
        # if not jpeg file then skip over it (eg .DS_Store file)
        continue

    img_path = os.path.join(path_to_images, img)
    img_loaded = cv2.imread(img_path, cv2.IMREAD_COLOR)
    images[img] = img_loaded

# test image
img_name = 'im001-healthy.jpg'
image = images[img_name]

# check it has loaded
if not image is None:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # b&w

    # save processed image into Results directory under same filename
    cv2.imwrite(os.path.join('Results', img_name), image)
else:
    print("No image file was loaded.")


