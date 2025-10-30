# %%
import sys
import os

import cv2
import numpy as np

from criminisi_inpainter import Criminisi_Inpainter

try:
    path_to_images = sys.argv[1]  # path given as command line argument
except:
    # default path
    path_to_images = 'image_processing_files/xray_images'

# create results directory if doesnt already exist
if not os.path.isdir('Results'):
    os.mkdir('Results')

# delete contents of results directory if wanted
refreshResults = False
if refreshResults:
    for file in os.listdir('Results'):
        os.remove(os.path.join('Results', file))

# %%
# helper functions

def fix_perspective(image):
    height, width = image.shape[:2]

    mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # isolate image and missing region outlines
    mask = cv2.inRange(mask, 0, 10)

    # laplaciam edge detection
    mask = cv2.Laplacian(mask, cv2.CV_8U, ksize=7)

    # get straight edges (i.e. square xray image border)
    # extract contours (only need extreme outer contours)
    contours, _ = cv2.findContours(mask, mode=cv2.RETR_EXTERNAL,
                                   method=cv2.CHAIN_APPROX_SIMPLE)
    # method stores only contour endpoints rather than all points on contour

    # create mask for ROI (xray image) outline (square)
    square_mask = np.zeros((height, width), np.uint8)
    cv2.drawContours(square_mask, contours, -1, (255, 255, 255), 1)

    # get 4 corners of xray image using Shi-Tomasi method (through OpenCV)
    corners = cv2.goodFeaturesToTrack(square_mask, 4, 0.01, 10)
    corners = corners.reshape(4, 2)

    border_width = 4  # border around image (set = 0 for no border)

    # desired 4 corners after warp
    map_to = [[border_width, border_width],
              [width-border_width, border_width],
              [border_width, height-border_width],
              [width-border_width, height-border_width]]
    map_from = [[0., 0.]] * 4  # to store 4 corners of ROI

    # sort corners of ROI to align with corners of frame
    # such that map_from[i] is closest corner to map_to[i]
    for i in range(len(map_to)):  # iterate through map_to
        min_dist = float('inf')
        # for each corner found by Shi-Tomasi:
        for c in corners:
            # use numpy to get euclidean distance (l2 norm) between 2 points
            dist = np.linalg.norm(c - map_to[i])
            if dist < min_dist:
                # update closest corner to map_to[i]
                min_dist = dist
                map_from[i] = c  # i.e. c is closest corner to map_to[i] found

    # shift perspective using perspective transform
    M = cv2.getPerspectiveTransform(
        np.array(map_from, np.float32), np.array(map_to, np.float32))
    image = cv2.warpPerspective(image, M, (width, height))

    return image

# %%
# Main function

def process_image(image):
    '''
    Process xray image. Keeps same resolution.
    '''
    # STEP 1 - unwarp perspective ===========================
    image = fix_perspective(image)

    # STEP 2 - inpaint ====================================
    # greyscale
    grey_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # create mask of missing region for inpainting
    inpainting_mask = cv2.inRange(grey_img, 0, 10)

    # add 10-pixel-wide black edge to mask
    # (dont inpaint image's existing black border)
    border_width = 10
    inpainting_mask[:border_width, :] = 0
    inpainting_mask[-border_width:, :] = 0
    inpainting_mask[:, :border_width] = 0
    inpainting_mask[:, -border_width:] = 0

    # dilate mask to expand missing region
    # (this overlaps region to be inpainted with known region, so more seamless inpainting)
    dilation_kernel = np.ones((4, 3), np.uint8)
    inpainting_mask = cv2.dilate(
        inpainting_mask, dilation_kernel, iterations=1)

    # turn into binary mask
    inpainting_mask[inpainting_mask > 0] = 1

    inpainter = Criminisi_Inpainter(image, inpainting_mask, patch_size=9)
    image = inpainter.inpaint()

    # STEP 3 - noise filtering ==============================

    # remove s&p with median filter
    image = cv2.medianBlur(image, 3)

    # nl means (converts to lab for filtering)
    # h corresponds to filter strength in l channel
    # hColor corresponds to filter strength in a and b channels
    image = cv2.fastNlMeansDenoisingColored(
        image, None, h=7, hColor=1,
        templateWindowSize=9, searchWindowSize=31)

    # sharpen edges
    lap_kernel = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ])
    laplacian = cv2.filter2D(image, cv2.CV_8U, lap_kernel)
    image = cv2.subtract(image, laplacian)

    # STEP 4 - Colour and contrast ==========================
    # RGB --> LAB
    # l = lightness, a = green-->red, b = blue-->yellow
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_image)

    # hist eq on l channel
    tile_size = 4
    clahe = cv2.createCLAHE(clipLimit=3,  # best between 2.3-3.2
                            tileGridSize=(tile_size, tile_size))
    l = clahe.apply(l)

    # gamma correction
    gamma = 0.7
    l = np.clip(np.power(l / 255.0, gamma) * 255.0, 0, 255).astype(np.uint8)

    # LAB to BGR
    lab_image = cv2.merge((l, a, b))
    image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)

    return image

# %%
# MAIN LOOP

# set start point
start_from = 'im044-healthy.jpg'

# load images & process
for tag in os.listdir(path_to_images):
    if not tag.endswith('.jpg'):
        # if not jpeg file then skip over it (eg .DS_Store file)
        continue

    # skip tags up to start_from. start_from = None to not skip any.
    if tag != start_from and start_from != None:
        continue
    start_from = None

    img_path = os.path.join(path_to_images, tag)
    img_loaded = cv2.imread(img_path, cv2.IMREAD_COLOR)

    # check it has loaded
    if not img_loaded is None:
        processed = process_image(img_loaded)
        cv2.imwrite(os.path.join('Results', tag), processed)
        # print('Successfully processed ' + tag + '!')
    else:
        print(tag + " failed to load.")
