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

refreshResults = False

# delete contents of results directory if wanted
if refreshResults:
    for file in os.listdir('Results'):
        os.remove(os.path.join('Results', file))

def get_circle_square_masks(image):
    height, width = image.shape[:2]

    mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # isolate black circle by thresholding
    mask = cv2.inRange(mask, 0, 10)

    # laplaciam edge detection
    mask = cv2.Laplacian(mask, cv2.CV_8U, ksize=7)
    # mask = cv2.Canny(mask, 100, 300)

    # extract contours (only need extreme outer contours)
    contours, _ = cv2.findContours(mask, mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_SIMPLE)
    # method stores only contour endpoints rather than all points on contour

    # create seperate masks for xray image outline (square) and missing region outline (circle)
    square_mask = np.zeros((height, width), np.uint8)
    circle_mask = mask.copy()

    cv2.drawContours(square_mask, contours, -1, (255, 255, 255), 1)
    cv2.drawContours(circle_mask, contours, -1, (0, 0, 0), 1)

    return circle_mask, square_mask

def fix_perspective(image, square_mask):
    height, width = image.shape[:2]

    # get 4 corners of xray image
    corners = cv2.goodFeaturesToTrack(square_mask, 4, 0.01, 10)
    corners = corners.reshape(4, 2)

    map_to = [[0, 0], [width, 0], [0, height], [width, height]]  # 4 corners of new image
    map_from = [[0., 0.]] * 4  # to store 4 corners of ROI

    # sort corners of ROI to align with corners of frame
    for i in range(len(map_to)):
        min_dist = 1000000
        for c in corners:
            dist = np.linalg.norm(c - map_to[i])
            if dist < min_dist:
                min_dist = dist
                map_from[i] = c

    # put coloured circles on object corners on image
    # corners = corners.astype(np.int32)  # convert to integers
    # colours = [(0, 255, 255), (0, 255, 0), (0, 0, 255), (255, 255, 0)] # yellow, green, red, cyan
    # for i in range(len(corners)):
    #     x, y = corners[i].ravel()
    #     cv2.circle(image, (x, y), 3, colours[i], -1)
    
    # shift perspective
    M = cv2.getPerspectiveTransform(np.array(map_from, np.float32), np.array(map_to, np.float32))
    image = cv2.warpPerspective(image, M, (width, height))

    return image

def remove_R(image):
    height, width = image.shape[:2]

    # create template containing red 'R'
    template_size = 50
    template = np.zeros((template_size, template_size, 3), np.uint8)
    cv2.putText(template, 'R', (template_size // 2, template_size // 2), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), thickness = 1)

    return template
    # match template to image
    corr = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(corr)
    top_left = max_loc
    bottom_right = (top_left[0] + width, top_left[1] + height)

    # remove 'R' from image
    image = cv2.rectangle(image, top_left, bottom_right, (0, 0, 0), 1)

    return image

def conservative_smoothing(image):
    # from linked towards data science article
    temp = []
    indexer = filter_size // 2
    new_image

def process_image(image):
    height, width = image.shape[:2]

    # create seperate masks for xray image outline (square) and missing region outline (circle)
    _, square_mask = get_circle_square_masks(image)

    # fix warped perspective
    image = fix_perspective(image, square_mask)

    # histogram equalisation
    # convert to hsv colour space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # apply histogram equalisation to V channel
    image[:, :, 2] = cv2.equalizeHist(image[:, :, 2])

    # convert back to BGR
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    # create mask using thresholding for missing circle region
    thresh = 60
    mask = cv2.inRange(image, (0, 0, 0), (thresh, thresh, thresh))
    
    # inpaint using circle mask
    image = cv2.inpaint(image, mask, 9, cv2.INPAINT_NS)

    # remove red 'R' from image
    # image = remove_R(image)

    # remove salt and pepper noise with conservative smoothing


    # greyscale
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # non local means filtering (https://docs.opencv.org/3.4/d5/d69/tutorial_py_non_local_means.html)
    # image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    # make brighter (copilot)
    # gain = 1.2
    # bias = 0
    # image = cv2.convertScaleAbs(image, alpha = gain, beta = bias)

    # false colour mapping to greyscale
    # image = cv2.applyColorMap(image, cv2.COLORMAP_JET)

    return image

# TESTING ======================================================================
isTesting = True

healthy = 'im001-healthy.jpg'
pneumonia = 'im053-pneumonia.jpg'

img_name = 'im100-pneumonia.jpg'
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
