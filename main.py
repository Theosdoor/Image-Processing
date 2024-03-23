import sys
import os

import cv2
import numpy as np

try:
    path_to_images = sys.argv[1] # path given as command line argument
    # TODO check if path is valid like in ai search
except:
    path_to_images = 'image_processing_files/xray_images'

# create results directory if doesnt already exist
if not os.path.isdir('Results'):
    os.mkdir('Results')

isTesting = True
refreshResults = True

# delete contents of results directory if wanted
if refreshResults:
    for file in os.listdir('Results'):
        os.remove(os.path.join('Results', file))

def process_image(image):
    # greyscale
    grey_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # find black circle in image



    # non local means filtering (copilot, from https://docs.opencv.org/3.4/d5/d69/tutorial_py_non_local_means.html)
    # image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    # hisogram equalisation (copilot)
    # convert to hsv colour space
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # apply histogram equalisation to the V channel
    # image[:, :, 2] = cv2.equalizeHist(image[:, :, 2])


    # convert back to BGR
    # image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    # make brighter (copilot)
    # gain = 1.2
    # bias = 0
    # image = cv2.convertScaleAbs(image, alpha = gain, beta = bias)

    # false colour mapping to greyscale
    image = cv2.applyColorMap(grey_img, cv2.COLORMAP_JET)

    return image

# TESTING ======================================================================
# test image
healthy = 'im001-healthy.jpg'
pneumonia = 'im053-pneumonia.jpg'

img_name = healthy
# TESTING ======================================================================

# load images & process
for tag in os.listdir(path_to_images):
    if not tag.endswith('.jpg'):
        # if not jpeg file then skip over it (eg .DS_Store file)
        continue

    # FOR TEST
    if tag != img_name and isTesting:
        continue

    img_path = os.path.join(path_to_images, tag)
    img_loaded = cv2.imread(img_path, cv2.IMREAD_COLOR)

    # check it has loaded
    if not img_loaded is None:
        processed = process_image(img_loaded)
        cv2.imwrite(os.path.join('Results', tag), processed)
    else:
        print(tag + " failed to load.")
