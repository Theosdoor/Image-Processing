import os

import cv2
import numpy as np

path_to_images = 'image_processing_files/xray_images'

def get_circle_square_masks(image):
    '''
    create seperate masks for xray image outline (square) and missing region outline (circle)
    '''

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

    # create seperate masks for xray image outline (square) and missing region outline (circle)
    square_mask = np.zeros((height, width), np.uint8)
    circle_mask = mask.copy()

    cv2.drawContours(square_mask, contours, -1, (255, 255, 255), 1)
    cv2.drawContours(circle_mask, contours, -1, (0, 0, 0), -1)

    return circle_mask, square_mask

def show_circles(image):
    # use HoughCircles to find circles in image (opencv docs)
    # find black circle in image
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grey = cv2.medianBlur(grey, 5) # make edges clear!

    rows = grey.shape[0]
    circles = cv2.HoughCircles(grey, cv2.HOUGH_GRADIENT, 1, rows / 8, param1=100, param2=30, minRadius=3, maxRadius=20)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:  # Convert circles[0, :] to a list
            center = (i[0], i[1]) # center
            cv2.circle(image, center, 1, (0, 100, 100), 3) # outline
            radius = i[2]
            cv2.circle(image, center, radius, (255, 0, 255), 3)
    else:
        exit("No circles found in image.")

    # cv2.imwrite('detected circles', image)

    return image

def match_circles(image):
    # define size of circle
    width = image.shape[1] // 8
    height = image.shape[0] // 8
    radius = width // 2

    # create black circle template
    black_circle = np.ones((height, width, 3), np.uint8)
    cv2.circle(black_circle, (int(width / 2), int(height / 2)),
               radius, (0, 0, 0), -1)
    
    # match template to image
    corr = cv2.matchTemplate(image, black_circle, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(corr)
    top_left = max_loc
    bottom_right = (top_left[0] + width, top_left[1] + height)
    # cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
    cv2.circle(image, (top_left[0] + radius, top_left[1] + radius), radius, (0, 255, 0), 2)


    # cv2.imshow('black circle', black_circle)
    # cv2.imshow('matched circles', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return image

def process_image(image):
    # greyscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rows, cols = image.shape
    # # define size of circle
    # width = image.shape[1]
    # height = image.shape[0]
    # radius = width // 2

    # # create black circle mask
    # black_circle = np.ones((height, width, 3), np.uint8)
    # cv2.circle(black_circle, (int(width / 2), int(height / 2)),
    #            radius, (0, 0, 0), -1)

    # # pad image with border
    # border_size = 6
    # image = cv2.copyMakeBorder(image, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    mask = image.copy()

    # isolate black circle by thresholding
    mask = cv2.inRange(mask, 0, 10)

    # laplaciam edge detection
    mask = cv2.Laplacian(mask, cv2.CV_8U, ksize=7)
    # mask = cv2.Canny(mask, 100, 300)

    # extract contours to get seperate mask of circle and square
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    square_mask = np.zeros((rows, cols), np.uint8)
    circle_mask = mask.copy()

    cv2.drawContours(square_mask, contours, -1, (255, 255, 255), 1)
    cv2.drawContours(circle_mask, contours, -1, (0, 0, 0), 1)

    # get 4 corners of black square
    corners = cv2.goodFeaturesToTrack(square_mask, 4, 0.01, 10)
    corners = corners.reshape(4, 2)

    map_to = [[0, 0], [cols, 0], [0, rows], [cols, rows]]  # 4 corners of new image
    map_from = [[0, 0]] * 4  # to store 4 corners of ROI

    # sort corners of ROI to align with corners of frame
    for i in range(len(map_to)):
        min_dist = 1000000
        for c in corners:
            dist = np.linalg.norm(c - map_to[i])
            if dist < min_dist:
                min_dist = dist
                map_from[i] = c

    # put coloured circles on object corners on image
    # image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # corners = corners.astype(np.int32)  # convert to integers
    # colours = [(0, 255, 255), (0, 255, 0), (0, 0, 255), (255, 255, 0)] # yellow, green, red, cyan
    # for i in range(len(corners)):
    #     x, y = corners[i].ravel()
    #     cv2.circle(image, (x, y), 3, colours[i], -1)

    # shift perspective
    M = cv2.getPerspectiveTransform(np.array(map_from, np.float32), np.array(map_to, np.float32))
    image = cv2.warpPerspective(image, M, (cols, rows))

    # # apply fft to mask
    # rows, cols = mask.shape
    # crow, ccol = rows // 2, cols // 2 # center of mask

    # fft = np.fft.fft2(mask)
    # fft_shifted = np.fft.fftshift(fft)

    # # correct orientation using magnitude spectrum
    # magnitude_spectrum = 20 * np.log(np.abs(fft_shifted))
    # magnitude_spectrum = cv2.inRange(magnitude_spectrum, 0, 10)


    # # apply low pass filtering
    # # radius = 30
    # # fft_shifted[crow-radius : crow+radius, ccol-radius : ccol+radius] = 0 # low pass filter

    # # shift back
    # fft_inv_shift = np.fft.ifftshift(fft_shifted)
    # mask = np.fft.ifft2(fft_inv_shift)
    # mask = np.abs(mask) # get magnitude


    # image inpainting
    # image = cv2.inpaint(image, mask, inpaintRadius=6, flags=cv2.INPAINT_NS)

    # image = match_circles(image)

    # hisogram equalisation (copilot)
    # convert to hsv colour space
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # # apply histogram equalisation to the V channel
    # image[:, :, 2] = cv2.equalizeHist(image[:, :, 2])


    # # convert back to BGR
    # image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    # nlm filtering
    # image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    # # make brighter (copilot)
    # gain = 1.2
    # bias = 0
    # image = cv2.convertScaleAbs(image, alpha = gain, beta = bias)


    # false colour mapping to greyscale
    # image = cv2.applyColorMap(image, cv2.COLORMAP_JET)

    # create template containing red 'R'
    template_size = 100
    template = np.zeros((template_size, template_size, 3), np.uint8)
    cv2.putText(template, 'R', (template_size // 2, template_size // 2), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), thickness = 1)

    mask = template

    cv2.imwrite(os.path.join('Results', 'mask.png'), mask)
    return image


# TESTING ======================================================================
# test image
isTesting = True
healthy = 'im001-healthy.jpg'
healthy5 = 'im005-healthy.jpg'
edge_healthy = 'im014-healthy.jpg'
pneumonia = 'im054-pneumonia.jpg'

img_name = healthy5
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
