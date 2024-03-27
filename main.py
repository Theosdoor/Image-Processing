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

def fix_perspective(image):
    height, width = image.shape[:2]

    mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # isolate image and missing region outlines
    mask = cv2.inRange(mask, 0, 10)

    # laplaciam edge detection
    mask = cv2.Laplacian(mask, cv2.CV_8U, ksize=7)
    # mask = cv2.Canny(mask, 100, 300)

    # extract contours (only need extreme outer contours)
    contours, _ = cv2.findContours(mask, mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_SIMPLE)
    # method stores only contour endpoints rather than all points on contour

    # create mask for xray image outline (square)
    square_mask = np.zeros((height, width), np.uint8)
    cv2.drawContours(square_mask, contours, -1, (255, 255, 255), 1)

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

def conservative_smoothing(image, kernel_size):
    '''
    from https://towardsdatascience.com/image-filters-in-python-26ee938e57d2 

    applies to greyscale image
    '''
    temp = []
    indexer = kernel_size // 2 # keep track of distance either side of center pixel
    new_image = image.copy()
    nrow, ncol = image.shape
    
    for i in range(nrow): # for each pixel in row
        for j in range(ncol): # for each pixel in col
            for k in range(i-indexer, i+indexer+1): # for each pixel in kernel row (L-->R)
                for m in range(j-indexer, j+indexer+1): # for each pixel in kernel col (T-->B)
                    if (k > -1) and (k < nrow): # check in bounds of image
                        if (m > -1) and (m < ncol): # check in bounds of image
                            temp.append(image[k,m]) # add to matrix of pixel values in neighbourhood
            temp.remove(image[i,j]) # ignore center pixel value
            
            max_value = max(temp)
            min_value = min(temp)
            if image[i,j] > max_value:
                new_image[i,j] = max_value
            elif image[i,j] < min_value:
                new_image[i,j] = min_value
            temp =[]
    
    return new_image.copy()

def colour_transfer(src, dst):
    '''
    from https://pyimagesearch.com/2014/06/30/super-fast-color-transfer-images/ 

    src = image to take colour stats from
    dst = image to apply statistics to
    '''
    # convert images to L*a*b* colour space
    src = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype('float32')
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2LAB).astype('float32')

    # compute colour statistics - arrays of mean and std for each channel
    src_mean, src_std = cv2.meanStdDev(src)
    dst_mean, dst_std = cv2.meanStdDev(dst)

    # subtract the means from the target image
    (l, a, b) = cv2.split(dst)
    l -= dst_mean[0]
    a -= dst_mean[1]
    b -= dst_mean[2]

    # scale by the standard deviations
    l = (l * (src_std[0] / dst_std[0]))
    a = (a * (src_std[1] / dst_std[1]))
    b = (b * (src_std[2] / dst_std[2]))

    # add in the source mean
    l += src_mean[0]
    a += src_mean[1]
    b += src_mean[2]

    # clip the pixel intensities to [0, 255] if they fall outside this range
    l = np.clip(l, 0, 255)
    a = np.clip(a, 0, 255)
    b = np.clip(b, 0, 255)

    # merge channels back together and convert back to BGR
    transfer = cv2.merge([l, a, b])
    transfer = cv2.cvtColor(transfer.astype('uint8'), cv2.COLOR_LAB2BGR)

    return transfer

def ciminisi_compute_confidence(src, target):
    pass

def criminisi_inpaint(src, target, patch_size = 9, search_size = 15, max_iter = 1000):
    '''
    from https://ieeexplore.ieee.org/document/1323101

    INPUTS:
    =======
    src (Phi) = source region for inpainting
    target (Omega) = region to be removed and filled
    '''
    height, width = src.shape[:2]

    # confidence values for each pixel
    C = np.zeros((height, width), np.uint8)
    # data values for each pixel
    D = np.zeros((height, width), np.uint8)
    #  priority values for each pixel
    P = np.zeros((height, width), np.uint8)

    # initialise confidence values: 1 for missing region, 0 for known region
    for i in range(height):
        for j in range(width):
            # C = 1 if pixel not in target region
            if target[i, j] == 0:
                C[i, j] = 1
    
    # repeat until region filled
    while True and max_iter > 0:
        max_iter -= 1

        # identify fill front
        fill_front = cv2.Canny(target, 100, 200)

        # if fill front empty, break
        if np.sum(fill_front) == 0:
            break

        # computer priorities for each pixel in fill front
        for i in range(height):
            for j in range(width):
                # if pixel in fill front
                if fill_front[i, j] != 0:
                    # compute priority
                    P[i, j] = C[i, j] * D[i, j]






    return src

def process_image(image):
    '''
    Process xray image. Keeps same resolution.
    '''
    # fix warped perspective
    image = fix_perspective(image)

    # greyscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # create mask using thresholding for missing circle region
    # mask = cv2.inRange(image, 30, 230)
    # mask = cv2.bitwise_not(mask)
    mask = cv2.inRange(image, 0, 10)

    # inpaint using circle mask
    # image = cv2.inpaint(image, mask, 9, cv2.INPAINT_NS)
    image = criminisi_inpaint(image, mask)
    return image

    # remove red 'R' from image
    # image = remove_R(image)

    # remove s&p noise
    # image = conservative_smoothing(image, 5)
    # image = cv2.medianBlur(image, 3)

    # non local means filtering (https://docs.opencv.org/3.4/d5/d69/tutorial_py_non_local_means.html)
    image = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
    # image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    # clahe
    # clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    # image = clahe.apply(image)

    # make brighter (copilot)
    # gain = 1.2
    # bias = 0
    # image = cv2.convertScaleAbs(image, alpha = gain, beta = bias)

    # add false colour
    # image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    # image = colour_transfer(temp, image)

    # histogram equalisation in l*a*b* colour space
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # l, a, b = cv2.split(image)
    # # l = cv2.equalizeHist(l)
    # clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    # l = clahe.apply(l)
    # image = cv2.merge([l, a, b])
    # image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)

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
