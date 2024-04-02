import sys
import os

import cv2
import numpy as np
import math

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

# helper functions
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

def colour_map(grey_img):
    pass

def criminisi_compute_confidence(src, target):
    pass

def criminisi_inpaint(src, target, patch_size = 9, search_size = 15, max_iter = 1000):
    '''
    from https://ieeexplore.ieee.org/document/1323101

    INPUTS:
    =======
    src (Phi) = source region for inpainting
    target (Omega) = region to be removed and filled.
        Binary mask, same size as src, with region to be filled = 1, 0 elsewhere
    '''
    height, width = src.shape[:2]
    mask = target.copy()

    # confidence values for each pixel
    # initialised at 1 for missing region, 0 for known region
    # i.e. 1 - binary target mask
    C = (1-target).astype(np.float32)
    # data values for each pixel
    D = np.zeros((height, width), np.uint8)
    # priority values for each pixel
    P = np.zeros((height, width), np.uint8)
    
    # repeat until region filled
    for it in range(max_iter):
        # identify fill front
        # fill_front = cv2.Canny(target, 100, 200)
        fill_front = cv2.Laplacian(mask, cv2.CV_8U, ksize=3)

        # if fill front empty, break
        if np.sum(fill_front) == 0:
            break

        # update priorities for each pixel in fill front
        for i in range(height):
            for j in range(width):
                # if pixel in fill front
                if fill_front[i, j] != 0:
                    # compute priority
                    P[i, j] = C[i, j] * D[i, j]






    return src

def create_band_pass_filter(width, height, radius_small, radius_big):
    '''
    from amir github
    '''
    assert(radius_big > radius_small)

    bp_filter = np.ones((height, width, 2), np.float32)
    cv2.circle(bp_filter, (int(width / 2), int(height / 2)),
               radius_big, (0, 0, 0), thickness=-1)

    cv2.circle(bp_filter, (int(width / 2), int(height / 2)),
               radius_small, (1, 1, 1), thickness=-1)

    return bp_filter

def create_butterworth_high_pass_filter(width, height, d, n):
    hp_filter = np.zeros((height, width, 2), np.float32)
    centre = (width / 2, height / 2)

    for i in range(0, hp_filter.shape[1]):  # image width
        for j in range(0, hp_filter.shape[0]):  # image height
            radius = max(1, math.sqrt(math.pow((i - centre[0]), 2.0) + math.pow((j - centre[1]), 2.0)))
            hp_filter[j, i] = 1 / (1 + math.pow((d / radius), (2 * n)))
    return hp_filter

def create_butterworth_low_pass_filter(width, height, d, n):
    lp_filter = np.zeros((height, width, 2), np.float32)
    centre = (width / 2, height / 2)

    for i in range(0, lp_filter.shape[1]):  # image width
        for j in range(0, lp_filter.shape[0]):  # image height
            radius = max(1, math.sqrt(math.pow((i - centre[0]), 2.0) + math.pow((j - centre[1]), 2.0)))
            lp_filter[j, i] = 1 / (1 + math.pow((radius / d), (2 * n)))
    return lp_filter

def create_butterworth_bp_filter(width, height, n, d0, d1):
    bp_filter = np.zeros((height, width, 2), np.float32)
    centre = (width / 2, height / 2)

    for i in range(0, bp_filter.shape[1]):  # image width
        for j in range(0, bp_filter.shape[0]):  # image height
            break

    return bp_filter

def band_pass(grey_img, radius_small, radius_big):
    '''
    from amir github
    '''
    assert(radius_big > radius_small)
    height, width = grey_img.shape[:2]

    # set up optimized DFT settings
    nheight = cv2.getOptimalDFTSize(height)
    nwidth = cv2.getOptimalDFTSize(width)

    # perform the DFT and get complex output
    dft = cv2.dft(np.float32(grey_img), flags=cv2.DFT_COMPLEX_OUTPUT)

    dft_shifted = np.fft.fftshift(dft)

    # do the filtering
    bp_filter = create_band_pass_filter(nwidth, nheight, radius_small, radius_big)

    dft_filtered = cv2.mulSpectrums(dft_shifted, bp_filter, flags=0)

    inv_dft = np.fft.fftshift(dft_filtered)

    filtered_img = cv2.dft(inv_dft, flags=cv2.DFT_INVERSE)

    # normalized the filtered image into 0 -> 255 (8-bit grayscale) so we
    # can see the output
    min_val, max_val, min_loc, max_loc = \
        cv2.minMaxLoc(filtered_img[:, :, 0])
    filtered_img_normalised = filtered_img[:, :, 0] * (
        1.0 / (max_val - min_val)) + ((-min_val) / (max_val - min_val))
    filtered_img_normalised = np.uint8(filtered_img_normalised * 255)

    # calculate the magnitude spectrum and log transform + scale it for
    # visualization
    magnitude_spectrum = cv2.magnitude(dft_filtered[:, :, 0], dft_filtered[:, :, 1])
    magnitude_spectrum += .4 # avoid log(0)
    magnitude_spectrum = np.log(magnitude_spectrum)
    
    # create a 8-bit image to put the magnitude spectrum into
    magnitude_spectrum_normalised = np.zeros((nheight, nwidth, 1), np.uint8)

    # normalized the magnitude spectrum into 0 -> 255 (8-bit grayscale) so
    # we can see the output
    cv2.normalize(
            np.uint8(magnitude_spectrum),
            magnitude_spectrum_normalised,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX)

    return filtered_img_normalised, magnitude_spectrum_normalised

# Main function
def process_image(image):
    '''
    Process xray image. Keeps same resolution.
    '''
    # fix warped perspective
    image = fix_perspective(image)

    # histogram equalisation in l*a*b* colour space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(image)

    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    # l = clahe.apply(l)

    image = cv2.merge([l, a, b])
    image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)

    # greyscale
    grey_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # create mask using thresholding for missing circle region
    # mask = cv2.inRange(image, 30, 230)
    # mask = cv2.bitwise_not(mask)
    # grey_mask = cv2.inRange(grey_img, 0, 10)
    colour_mask = cv2.inRange(image, (0, 0, 0), (10, 10, 10))

    # inpaint using circle mask
    # grey_img = cv2.inpaint(grey_img, grey_mask, 9, cv2.INPAINT_NS)
    # image = cv2.inpaint(image, colour_mask, 9, cv2.INPAINT_NS)
    image = criminisi_inpaint(image, colour_mask)
    
    # remove s&p noise
    # grey_img = conservative_smoothing(grey_img, 5)
    # grey_img = cv2.medianBlur(grey_img, 3)


    # band pass filter
    # grey_img, magnitude_spectrum = band_pass(grey_img, 1, 200)

    # remove red 'R' from image
    # image = remove_R(image)

    # non local means filtering (https://docs.opencv.org/3.4/d5/d69/tutorial_py_non_local_means.html)
    # grey_img = cv2.fastNlMeansDenoising(grey_img, None, 10, 7, 21)
    # image = cv2.fastNlMeansDenoisingColored(image, None, h=8, hColor=10, templateWindowSize=7, searchWindowSize=21)
    # use greyscale nlm on different colour channels w different h values

    # add false colour
    # colour_mapped = cv2.applyColorMap(grey_img, cv2.COLORMAP_JET)
    # image = colour_transfer(image, colour_mapped)
    # image = grey_img

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
