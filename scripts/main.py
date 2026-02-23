# %%
import sys
import os

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import process_image

try:
    path_to_images = sys.argv[1]  # path given as command line argument
except IndexError:
    # default path
    path_to_images = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'xray_images')

# global variables
results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')

# create results directory if doesnt already exist
if not os.path.isdir(results_dir):
    os.mkdir(results_dir)

# delete contents of results directory if wanted
refreshResults = False
if refreshResults:
    for file in os.listdir(results_dir):
        os.remove(os.path.join(results_dir, file))




# %%
# MAIN LOOP

# set start point
start_from = None

# collect jpeg files to process
tags = [f for f in os.listdir(path_to_images) if f.endswith('.jpg')]

# load images & process
for tag in tqdm(tags, desc='Processing images'):
    # skip tags up to start_from. start_from = None to not skip any.
    if start_from is not None:
        if tag == start_from:
            start_from = None
        else:
            continue

    img_path = os.path.join(path_to_images, tag)
    img_loaded = cv2.imread(img_path, cv2.IMREAD_COLOR)

    # check it has loaded
    if img_loaded is not None:
        processed = process_image(img_loaded)
        cv2.imwrite(os.path.join(results_dir, tag), processed)
    else:
        tqdm.write(tag + " failed to load.")