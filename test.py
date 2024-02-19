import os

import cv2
import numpy as np

path_to_images = 'image_processing_files/xray_images'

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
    rows, cols, ch = image.shape
    # # define size of circle
    # width = image.shape[1]
    # height = image.shape[0]
    # radius = width // 2

    # # create black circle mask
    # black_circle = np.ones((height, width, 3), np.uint8)
    # cv2.circle(black_circle, (int(width / 2), int(height / 2)),
    #            radius, (0, 0, 0), -1)

    # pad image with border
    border_size = 6
    image = cv2.copyMakeBorder(image, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # isolate black circle by thresholding
    mask = cv2.inRange(mask, 0, 10)

    # sharpen mask edges using laplacian
    mask = cv2.Laplacian(mask, cv2.CV_8U)

    # get 4 corners of black square
    corners = cv2.goodFeaturesToTrack(mask, 4, 0.01, 10)
    corners = corners.reshape(4, 2)

    # get centers
    print(corners)
    # coord = (xmin, ymin, xmax, ymax)
    coord = (corners[0][1], corners[1][1], corners[2][0], corners[3][1])
    centerCoord = (coord[0]+(coord[2]/2), coord[1]+(coord[3]/2))
    print(centerCoord)
    print(mask.shape[0] / 2, mask.shape[1] / 2)

    xshift = mask.shape[0] / 2 - centerCoord[0]
    yshift = mask.shape[1] / 2 - centerCoord[1]

    shift = np.full(image.shape, [xshift, yshift, 0])

    # add shift to every element in image matrix
    image = np.add(image, shift)

    # put red circles on object corners
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    corners = corners.astype(np.int32)  # convert to integers
    for i in corners:
        x, y = i.ravel()
        cv2.circle(image, (x, y), 3, (0, 0, 255), -1)

    # put circle in middle of image
    cv2.circle(image, (int(image.shape[1] / 2), int(image.shape[0] / 2)), 3, (0, 0, 255), -1)

    # shift perspective
    pts1 = np.float32(corners)  # 4 corners of black square
    pts2 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])  # 4 corners of new image

    M = cv2.getPerspectiveTransform(pts1, pts2)

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
# exit("Finished.")

# test image
healthy = 'im003-healthy.jpg'
pneumonia = 'im053-pneumonia.jpg'

img_name = healthy

# load image
image = images[img_name]

# check it has loaded
if not image is None:
    image = process_image(image)

    # save processed image into Results directory under same filename
    cv2.imwrite(os.path.join('Results', img_name), image)
else:
    print("No image file was loaded.")
