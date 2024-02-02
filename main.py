import sys
import os

import cv2
import numpy as np

# path_to_images = '/home/theofarrell18/Assignments/Image-Processing/image_processing_files/xray_images'
path_to_images = sys.argv[1] # path given as command line argument
# check if valid path? like in ai search
for images in os.listdir(path_to_images):
    print(images)
