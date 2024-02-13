import sys
import os

import cv2
import numpy as np

# path_to_images = sys.argv[1] # path given as command line argument
# TODO check if valid path? like in ai search
path_to_images = 'image_processing_files/xray_images' # TODO REMOVE LATER to allow arg

# create results directory if doesnt already exist
if not os.path.isdir('Results'):
    os.mkdir('Results')

def process_image(image):
    # find black circle in image
    


    # non local means filtering (copilot, from https://docs.opencv.org/3.4/d5/d69/tutorial_py_non_local_means.html)
    # image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    # hisogram equalisation (copilot)
    # convert to hsv colour space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # apply histogram equalisation to the V channel
    image[:, :, 2] = cv2.equalizeHist(image[:, :, 2])


    # convert back to BGR
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    # make brighter (copilot)
    gain = 1.2
    bias = 0
    image = cv2.convertScaleAbs(image, alpha = gain, beta = bias)

    return image

# FOR TEST: dictionary matching image name to image matrix
images = {}

# load images & process
for tag in os.listdir(path_to_images):
    if not tag.endswith('.jpg'):
        # if not jpeg file then skip over it (eg .DS_Store file)
        continue

    img_path = os.path.join(path_to_images, tag)
    img_loaded = cv2.imread(img_path, cv2.IMREAD_COLOR)

    # FOR ALL IMAGES
    # processed = process_image(img_loaded)
    # cv2.imwrite(os.path.join('Results', tag), processed)

    # FOR TEST IMAGES
    images[tag] = img_loaded

# ==============================================================================
# TESTING (DELETE LATER)
# exit()

# test image
healthy = 'im001-healthy.jpg'
pneumonia = 'im053-pneumonia.jpg'

img_name = pneumonia

# load image
image = images[img_name]

# check it has loaded
if not image is None:
    image = process_image(image)

    # save processed image into Results directory under same filename
    cv2.imwrite(os.path.join('Results', img_name), image)
else:
    print("No image file was loaded.")
