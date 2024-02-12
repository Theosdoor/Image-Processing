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
    # non local means filtering (copilot, from https://docs.opencv.org/3.4/d5/d69/tutorial_py_non_local_means.html)
    # image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    # butterworth filter (copilot)

    # # separate out the channels
    # b, g, r = cv2.split(image)

    # # apply gaussian to each channel
    # b = cv2.GaussianBlur(b, (5, 5), 0)
    # g = cv2.GaussianBlur(g, (5, 5), 0)
    # r = cv2.GaussianBlur(r, (5, 5), 0)

    # # apply laplacian to each channel
    # b = cv2.Laplacian(b, cv2.CV_64F)
    # g = cv2.Laplacian(g, cv2.CV_64F)
    # r = cv2.Laplacian(r, cv2.CV_64F)

    # # recombine the channels
    # image = cv2.merge((b, g, r))

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

# dictionary matching image name to image matrix
images = {}

# load images & process
for tag in os.listdir(path_to_images):
    if not tag.endswith('.jpg'):
        # if not jpeg file then skip over it (eg .DS_Store file)
        continue

    img_path = os.path.join(path_to_images, tag)
    img_loaded = cv2.imread(img_path, cv2.IMREAD_COLOR)

    # processed = process_image(img_loaded)
    # cv2.imwrite(os.path.join('Results', tag), processed)
    images[tag] = img_loaded

# ==============================================================================
# TESTING (DELETE LATER)
# exit()

# test image
healthy = 'im001-healthy.jpg'
pneumonia = 'im053-pneumonia.jpg'

img_name = pneumonia
image = images[img_name]

# check it has loaded
if not image is None:
    image = process_image(image)

    # save processed image into Results directory under same filename
    cv2.imwrite(os.path.join('Results', img_name), image)
else:
    print("No image file was loaded.")


