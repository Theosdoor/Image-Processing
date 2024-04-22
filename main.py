import sys
import os

import cv2
import numpy as np
import math
import time
import matplotlib.pyplot as plt # TODO DELETE

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

    border_width = 4 # border around image (set = 0 for no border)

    # desired 4 corners after warp
    map_to = [[border_width, border_width],
              [width-border_width, border_width],
              [border_width, height-border_width],
              [width-border_width, height-border_width]]
    map_from = [[0., 0.]] * 4  # to store 4 corners of ROI

    # sort corners of ROI to align with corners of frame
    # such that map_from[i] is closest corner to map_to[i]
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

class Criminisi_Inpainter():
    '''
    From https://github.com/igorcmoura/inpaint-object-remover/blob/master/inpainter/inpainter.py.
    Adapted to use opencv rather than scipy and skimage.

    Requires binary mask of missing region, same size as image.
        Pixels values: 1 for missing, 0 for known region.
    '''
    def __init__(self, image, mask, patch_size=9, verbose=False, show_progress=False):
        self.image = image.astype('uint8') # TODO need to change to uint8? i think they are already that and need to keep same
        self.mask = mask.round().astype('uint8')
        self.patch_size = patch_size
        self.verbose = verbose
        self.show_progress = show_progress

        ## non-argument attributes
        self.iheight, self.iwidth = self.image.shape[:2] # TODO change var names to fit criminisi paper
        # The working image and working mask start as copies of the original
        # image and mask.
        self.working_image = np.copy(self.image)
        self.working_mask = np.copy(self.mask)
        self.front = np.zeros([self.iheight, self.iwidth])

        # The confidence is initially the inverse of the mask, that is, the
        # target region is 0 and source region is 1.
        self.confidence = (1 - self.mask).astype(float) # TODO check how astype(float) affects it
        # The data and priority matrices start with zero for all pixels.
        self.data = np.zeros([self.iheight, self.iwidth])
        self.priority = np.zeros([self.iheight, self.iwidth])

    def inpaint(self):
        '''
        Compute the new image and return it.
        '''
        self._validate_inputs()

        start_time = time.time()
        c = 0
        # loop until whole region is filled
        while True:
            self._find_front()

            if self.show_progress:
                name = 'Results/working_image_' + str(c) + '.png'
                cv2.imwrite(name, self.working_image)

            self._update_priority()

            target_pixel = self._find_highest_priority_pixel()
            find_start_time = time.time()
            source_patch = self._find_source_patch(target_pixel)
            if self.verbose:
                print('Time to find best: %f seconds' % (time.time()-find_start_time))

            self._update_image(target_pixel, source_patch)

            # check if finished filling
            if self._finished():
                break
            
            c += 1

        if self.verbose:
            print('Inpainting took %f seconds to complete' % (time.time() - start_time))
        return self.working_image

    def _validate_inputs(self):
        if self.image.shape[:2] != self.mask.shape:
            raise AttributeError('mask and image must be of the same size')

    def _find_front(self):
        """ 
        Find the fill front using laplacian on the mask.
        
        The laplacian gives mask edges - positive at the higher region (white)
        and negative at the lower region (black).
        We only want the the white region, which is inside the mask, so we
        filter the negative values.
        """
        self.front = cv2.Laplacian(self.working_mask, cv2.CV_8U, ksize=3)
        self.front[self.front < 0] = 0

    def _update_priority(self):
        self._update_confidence()
        self._update_data()
        # update priority for every pixel, P = C * D.
        # (multiply confidence and data matrix by fill front to apply to all
        #     pixels at once --> much more efficient than for loop!)
        self.priority = self.confidence * self.data * self.front

    def _update_confidence(self):
        new_confidence = np.copy(self.confidence)
        front_positions = np.argwhere(self.front > 0) # get list of pixels on fill front
        for point in front_positions:
            patch = self._get_patch(point)
            new_confidence[point[0], point[1]] = np.sum(np.sum(self._patch_data(self.confidence, patch))) / self.patch_size**2

        self.confidence = new_confidence

    def _update_data(self):
        normal = self._calc_normal_matrix()
        gradient = self._calc_gradient_matrix()

        normal_gradient = normal * gradient
        self.data = np.sqrt(
            normal_gradient[:, :, 0]**2 + normal_gradient[:, :, 1]**2
        ) + 0.001  # make sure data > 0

    def _calc_normal_matrix(self):
        x_kernel = np.array([[.25, 0, -.25], [.5, 0, -.5], [.25, 0, -.25]])
        y_kernel = np.array([[-.25, -.5, -.25], [0, 0, 0], [.25, .5, .25]])

        x_normal = cv2.filter2D(self.working_mask.astype(float), -1, x_kernel)
        y_normal = cv2.filter2D(self.working_mask.astype(float), -1, y_kernel)
        normal = np.dstack((x_normal, y_normal))

        height, width = normal.shape[:2]
        norm = np.sqrt(y_normal**2 + x_normal**2)
        norm = norm.reshape(height, width, 1) \
                 .repeat(2, axis=2)
        norm[norm == 0] = 1

        unit_normal = normal/norm
        return unit_normal

    def _calc_gradient_matrix(self):
        height, width = self.working_image.shape[:2]

        grey_image = cv2.cvtColor(self.working_image, cv2.COLOR_BGR2GRAY)
        grey_image[self.working_mask > 0] = 0

        gradient = np.nan_to_num(np.array(np.gradient(grey_image)))
        gradient_val = np.sqrt(gradient[0]**2 + gradient[1]**2)
        max_gradient = np.zeros([height, width, 2])

        front_positions = np.argwhere(self.front > 0)
        for p in front_positions:
            patch = self._get_patch(p)
            patch_y_gradient = self._patch_data(gradient[0], patch)
            patch_x_gradient = self._patch_data(gradient[1], patch)
            patch_gradient_val = self._patch_data(gradient_val, patch)

            patch_max_pos = np.unravel_index(
                patch_gradient_val.argmax(),
                patch_gradient_val.shape
            )

            max_gradient[p[0], p[1], 0] = patch_y_gradient[patch_max_pos]
            max_gradient[p[0], p[1], 1] = patch_x_gradient[patch_max_pos]

        return max_gradient

    def _find_highest_priority_pixel(self):
        point = np.unravel_index(self.priority.argmax(), self.priority.shape)
        return point

    def _find_source_patch(self, target_pixel):
        target_patch = self._get_patch(target_pixel)
        height, width = self.working_image.shape[:2]
        patch_height, patch_width = self._patch_shape(target_patch)

        best_match = None
        best_match_difference = 0

        lab_image = cv2.cvtColor(self.working_image, cv2.COLOR_BGR2LAB)

        for y in range(height - patch_height + 1):
            for x in range(width - patch_width + 1):
                source_patch = [
                    [y, y + patch_height-1],
                    [x, x + patch_width-1]
                ]
                if self._patch_data(self.working_mask, source_patch).sum() != 0:
                    continue

                difference = self._calc_patch_difference(
                    lab_image,
                    target_patch,
                    source_patch
                )

                if best_match is None or difference < best_match_difference:
                    best_match = source_patch
                    best_match_difference = difference
        return best_match

    def _update_image(self, target_pixel, source_patch):
        target_patch = self._get_patch(target_pixel)
        pixels_positions = np.argwhere(
            self._patch_data(
                self.working_mask,
                target_patch
            ) > 0
        ) + [target_patch[0][0], target_patch[1][0]]
        patch_confidence = self.confidence[target_pixel[0], target_pixel[1]]
        for point in pixels_positions:
            self.confidence[point[0], point[1]] = patch_confidence

        mask = self._patch_data(self.working_mask, target_patch)
        rgb_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        source_data = self._patch_data(self.working_image, source_patch)
        target_data = self._patch_data(self.working_image, target_patch)

        new_data = source_data*rgb_mask + target_data*(1-rgb_mask)

        self._copy_to_patch(
            self.working_image,
            target_patch,
            new_data
        )
        self._copy_to_patch(
            self.working_mask,
            target_patch,
            0
        )

    def _get_patch(self, point):
        half_patch_size = (self.patch_size-1)//2
        height, width = self.working_image.shape[:2]
        patch = [
            [
                max(0, point[0] - half_patch_size),
                min(point[0] + half_patch_size, height-1)
            ],
            [
                max(0, point[1] - half_patch_size),
                min(point[1] + half_patch_size, width-1)
            ]
        ]
        return patch

    def _calc_patch_difference(self, image, target_patch, source_patch):
        mask = 1 - self._patch_data(self.working_mask, target_patch)
        rgb_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        target_data = self._patch_data(
            image,
            target_patch
        ) * rgb_mask
        source_data = self._patch_data(
            image,
            source_patch
        ) * rgb_mask
        squared_distance = np.sum(((target_data - source_data)**2))
        euclidean_distance = np.sqrt(
            (target_patch[0][0] - source_patch[0][0])**2 +
            (target_patch[1][0] - source_patch[1][0])**2
        )  # tie-breaker factor
        return squared_distance + euclidean_distance

    def _finished(self):
        remaining = np.sum(self.working_mask)
        total = np.sum(self.mask)
        if self.verbose:
            print('%d of %d completed' % (total-remaining, total))
        return remaining == 0

    @staticmethod
    def _patch_shape(patch):
        return (1+patch[0][1]-patch[0][0]), (1+patch[1][1]-patch[1][0])

    @staticmethod
    def _patch_data(source, patch):
        return source[
            patch[0][0]:patch[0][1]+1,
            patch[1][0]:patch[1][1]+1
        ]

    @staticmethod
    def _copy_to_patch(dest, dest_patch, data):
        dest[
            dest_patch[0][0]:dest_patch[0][1]+1,
            dest_patch[1][0]:dest_patch[1][1]+1
        ] = data

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
    # bp_filter = create_band_pass_filter(nwidth, nheight, radius_small, radius_big)
    bp_filter = create_butterworth_low_pass_filter(nwidth, nheight, radius_big, 2)

    dft_filtered = cv2.mulSpectrums(dft_shifted, bp_filter, flags=0)

    inv_dft = np.fft.fftshift(dft_filtered)

    filtered_img = cv2.dft(inv_dft, flags=cv2.DFT_INVERSE)

    # normalized the filtered image into 0 -> 255 (8-bit grayscale) so we
    # can see the output
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(filtered_img[:, :, 0])
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
    # image = cv2.imread('sample-xray98.png', cv2.IMREAD_COLOR)
        
    # fix colour channel imbalance (green sections are white)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(image)

    # v, mag_spec = band_pass(v, 10, 50)

    # # plot hue
    # if isTesting:
    #     plt.figure()
    #     plt.grid()
    #     hist = cv2.calcHist([h], [0], None, [200], [0, 200])
    #     plt.plot(hist)
    #     plt.xlim([0, 180])
    #     plt.ylim([0, 7000])
    #     plt.xticks(np.arange(0, 180, 20), rotation=45)
    #     plt.savefig('histogram_hue.png')

    # # isolate green hue range
    # # hue_range = cv2.inRange(h, 43, 80)


    # # morph_close to join up green regions
    # # kernel = np.ones((5, 5), np.uint8)
    # # hue_range = cv2.morphologyEx(hue_range, cv2.MORPH_CLOSE, kernel, iterations=2)

    # # clip so all still green
    # # hue_range.clip(43, 80)

    # # v[hue_range == 0] = 0 # to see green elements only
    # # h[hue_range > 0] = 1

    # gamma = 0.9
    # lookUpTable = np.empty((1,256), np.uint8)
    # for i in range(256):
    #     lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    
    # # s[hue_range > 0] += 2 # so that it is not 0 or 1
    # # s = cv2.LUT(s, lookUpTable)

    image = cv2.merge((h, s, v))
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)



    # greyscale
    grey_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # bleached = cv2.inRange(grey_img, 215,225)

    # image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # h, s, v = cv2.split(image_hsv)

    # white_pix = np.argwhere(bleached > 0)

    # # get normal distribution between 40 and 80
    # mean = 60
    # std = 10
    # h[bleached > 0] = np.random.normal(mean, std, len(white_pix))

    # image = cv2.merge((h, s, v))
    # image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    # # plot hue
    # if isTesting:
    #     plt.figure()
    #     plt.grid()
    #     hist = cv2.calcHist([h], [0], None, [200], [0, 200])
    #     plt.plot(hist)
    #     plt.xlim([0, 180])
    #     plt.ylim([0, 7000])
    #     plt.xticks(np.arange(0, 180, 20), rotation=45)
    #     plt.savefig('hist_hue.png')

    # # return image

    # ===========================

    # create mask of missing region for inpainting
    inpainting_mask = cv2.inRange(grey_img, 0, 10)
    
    # add 10-pixel-wide black edge (dont inpaint black border)
    border_width = 10
    inpainting_mask[:border_width, :] = 0
    inpainting_mask[-border_width:, :] = 0
    inpainting_mask[:, :border_width] = 0
    inpainting_mask[:, -border_width:] = 0
    
    # dilate mask to expand missing region
    dilation_kernel = np.ones((4, 3), np.uint8)
    inpainting_mask = cv2.dilate(inpainting_mask, dilation_kernel, iterations=1)

    if isTesting:
        cv2.imwrite('inpainting_mask.png', inpainting_mask)

    # turn into binary mask
    inpainting_mask[inpainting_mask > 0] = 1

    if isTesting:
        verbose = True
    else:
        verbose = False

    inpainter = Criminisi_Inpainter(image, inpainting_mask, patch_size=9, verbose=verbose, show_progress=False)
    # image = inpainter.inpaint()
    image = cv2.inpaint(image, inpainting_mask, 3, cv2.INPAINT_TELEA)

    ## remove s&p
    image = cv2.medianBlur(image, 3)

    ## remove gaussian noise

    # nl means
    image = cv2.fastNlMeansDenoisingColored(
        image, None, h=7, hColor=1,
        templateWindowSize=9, searchWindowSize=31)

    # sharpen edges
    laplacian = cv2.Laplacian(image, cv2.CV_8U, ksize=1)
    image = cv2.subtract(image, laplacian)

    # bilateral filter
    # image = cv2.bilateralFilter(
    #     image, d=9, sigmaColor=75, sigmaSpace=75,
    #     borderType=cv2.BORDER_DEFAULT)

    return image

    # ===========================

    # convert to l*a*b* colour space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(image)

    gamma = 1.2
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    
    # l = cv2.LUT(l, lookUpTable)

    # hist eq on l channel
    tile_size = 4
    clahe = cv2.createCLAHE(clipLimit=2.6, # b/t 2.3-3.2
                            tileGridSize=(tile_size, tile_size))
    # l = clahe.apply(l)

    # put channels back together
    image = cv2.merge((l, a, b))

    # plot lab channels
    # plt.figure()
    # for i in range(3):
    #     hist = cv2.calcHist([image], [i], None, [256], [0, 256])
    #     plt.plot(hist)
    #     plt.xlim([1, 256])
    # plt.legend(['l', 'a', 'b'])
    # plt.savefig('histogram_lab.png')

    # convert back to BGR
    image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)

    b, g, r = cv2.split(image)

    tile_size = 4
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(tile_size, tile_size))
    # b = clahe.apply(b)
    # r = clahe.apply(r)

    image = cv2.merge((b, g, r))

    # show histogram of bgr channels
    # plt.figure()
    # colours = ['b', 'g', 'r']
    # for i in range(3):
    #     hist = cv2.calcHist([image], [i], None, [256], [1, 256])
    #     plt.plot(hist, color=colours[i])
    #     plt.xlim([1, 256])
    # plt.legend(['b', 'g', 'r'])
    # plt.savefig('histogram_rgb.png')

    return image

# TESTING ======================================================================
isTesting = False

healthy_1 = 'im001-healthy.jpg'
healthy_4 = 'im004-healthy.jpg' # example on gc
healthy_12 = 'im012-healthy.jpg' # small missing region
healthy_31 = 'im031-healthy.jpg' # sample image
healthy_42 = 'im042-healthy.jpg' # bad criminisi
pneumonia_53 = 'im053-pneumonia.jpg'
pneumonia_56 = 'im056-pneumonia.jpg' # bad contrast
pneumonia_74 = 'im074-pneumonia.jpg' # hard to see red R
pneumonia_98 = 'im098-pneumonia.jpg' # sample image
pneumonia_100 = 'im100-pneumonia.jpg'

img_name = pneumonia_98
# TESTING ======================================================================
start_time = time.time()

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
        process_start_time = time.time()
        processed = process_image(img_loaded)
        cv2.imwrite(os.path.join('Results', tag), processed)
        print(tag + " processed! Took %.2f seconds" % (time.time() - process_start_time))
    else:
        print(tag + " failed to load.")

print('Total time taken: %.2f seconds' % (time.time() - start_time))
