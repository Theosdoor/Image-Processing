import sys
import os

import cv2
import numpy as np
import math
import time

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
    nrow, ncol = image.shape[:2]
    
    for i in range(nrow): # for each pixel in row
        for j in range(ncol): # for each pixel in col
            for k in range(i-indexer, i+indexer+1): # for each pixel in kernel row (L-->R)
                for m in range(j-indexer, j+indexer+1): # for each pixel in kernel col (T-->B)
                    if (k >= 0) and (k < nrow): # check in bounds of image
                        if (m >= 0) and (m < ncol): # check in bounds of image
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

class Criminisi_Inpainter():
    '''
    From https://github.com/igorcmoura/inpaint-object-remover/blob/master/inpainter/inpainter.py.
    Adapted to use opencv rather than scipy and skimage.

    Requires binary mask of missing region, same size as image.
        Pixels values: 1 for missing, 0 for known region.
    '''
    def __init__(self, image, mask, patch_size=9, verbose=False, show_progress=False):
        self.image = image.astype('uint8')
        self.mask = mask.round().astype('uint8')
        self.patch_size = patch_size
        self.iheight, self.iwidth = self.image.shape[:2]
        self.verbose = verbose
        self.show_progress = show_progress

        # Non-initialized attributes
        self.working_image = np.copy(self.image)
        self.working_mask = np.copy(self.mask)
        self.front = np.zeros([self.iheight, self.iwidth])
        self.confidence = (1 - self.mask).astype(float)
        self.data = np.zeros([self.iheight, self.iwidth])
        self.priority = np.zeros([self.iheight, self.iwidth])

    def inpaint(self):
        '''
        Compute the new image and return it.
        '''
        self._validate_inputs()
        self._initialize_attributes()

        start_time = time.time()
        keep_going = True
        c = 0
        while keep_going:
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

            keep_going = not self._finished()
            c += 1

        if self.verbose:
            print('Inpainting took %f seconds to complete' % (time.time() - start_time))
        return self.working_image

    def _validate_inputs(self):
        if self.image.shape[:2] != self.mask.shape:
            raise AttributeError('mask and image must be of the same size')

    def _initialize_attributes(self):
        """ Initialize the non initialized attributes

        The confidence is initially the inverse of the mask, that is, the
        target region is 0 and source region is 1.

        The data starts with zero for all pixels.

        The working image and working mask start as copies of the original
        image and mask.
        """
        self.confidence = (1 - self.mask).astype(float)
        self.data = np.zeros([self.iheight, self.iwidth])

        self.working_image = np.copy(self.image)
        self.working_mask = np.copy(self.mask)

    def _find_front(self):
        """ Find the front using laplacian on the mask

        The laplacian will give us the edges of the mask, it will be positive
        at the higher region (white) and negative at the lower region (black).
        We only want the the white region, which is inside the mask, so we
        filter the negative values.
        """
        self.front = cv2.Laplacian(self.working_mask, cv2.CV_8U, ksize=3)
        self.front[self.front < 0] = 0

    def _update_priority(self):
        self._update_confidence()
        self._update_data()
        self.priority = self.confidence * self.data * self.front

    def _update_confidence(self):
        new_confidence = np.copy(self.confidence)
        front_positions = np.argwhere(self.front > 0)
        for point in front_positions:
            patch = self._get_patch(point)
            new_confidence[point[0], point[1]] = np.sum(np.sum(
                self._patch_data(self.confidence, patch))) / self.patch_size**2

        self.confidence = new_confidence

    def _update_data(self):
        normal = self._calc_normal_matrix()
        gradient = self._calc_gradient_matrix()

        normal_gradient = normal * gradient
        self.data = np.sqrt(
            normal_gradient[:, :, 0]**2 + normal_gradient[:, :, 1]**2
        ) + 0.001  # To be sure to have a greater than 0 data

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

def criminisi_inpaint(src, target, patch_size = 9, search_size = 15, max_iter = 1000):
    height, width = src.shape[:2]
    mask = target.copy()
    grey_img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # confidence values for each pixel
    # initialised at 1 for missing region, 0 for known region
    # i.e. 1 - binary target mask
    C = (1.0 - target).astype(np.float32)
    # data values for each pixel
    D = np.zeros((height, width), np.uint8)
    # priority values for each pixel
    P = np.zeros((height, width), np.uint8)
    
    # repeat until region filled
    for it in range(max_iter):
        # identify fill front and filter out negative values
        # fill_front = cv2.Canny(target, 100, 200)
        fill_front = cv2.Laplacian(mask, cv2.CV_8U, ksize=3)

        # if fill front empty, break
        if np.sum(fill_front) == 0:
            break

        # update priorities of patches centred on fill front
        front_positions = np.argwhere(fill_front > 0)
        for i in range(len(front_positions)):
            p = front_positions[i]
            # get patch centred at p
            # NOTE watchout for boundary patches?
            patch = [[p[0]-patch_size//2, p[1]-patch_size//2],
                        [p[0]+patch_size//2, p[1]+patch_size//2]]
            
            # update confidence values
            # C(p) = sum(confidence of pixels in patch) / patch area
            C[p[0], p[1]] = np.sum(C[p[0]-patch_size//2 : p[0]+patch_size//2+1,
                                     p[1]-patch_size//2 : p[1]+patch_size//2+1]) / patch_size**2
            
            # update data values
            # D(p) = magnitude(isophote at p * normal to ff at p) / alpha
            # NOTE use matrix mult to speed up?
            alpha = 255

            ## normal to fill front at p
            # imagine tangent line between two points either side of p on fill front
            if i == 0:
                preceding = front_positions[-1]
                successive = front_positions[i+1]
            elif i == len(front_positions) - 1:
                preceding = front_positions[i-1]
                successive = front_positions[0]
            else:
                preceding = front_positions[i-1]
                successive = front_positions[i+1]
            # gradient of normal = -1 / gradient of tangent
            normal_grad =  -(successive[1] - preceding[1]) / (successive[0] - preceding[0])
            # get unit vector normal through p
            normal = np.array([1, normal_grad])
            print(normal)
            exit()
            normal = normal / np.linalg.norm(normal)

            ## isophote at p = max image gradient in patch
            # get image gradient in patch
            

        break

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

    # greyscale
    grey_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # show histogram of bgr channels
    import matplotlib.pyplot as plt
    colours = ['b', 'g', 'r']
    for i in range(3):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=colours[i])
        plt.xlim([0, 256])
    plt.savefig('Results/histogram_1.png')

    # histogram equalisation in l*a*b* colour space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(image)

    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    l = clahe.apply(l)

    image = cv2.merge([l, a, b])
    image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)

    # remove s&p
    image = cv2.medianBlur(image, 3)

    # remove gaussian noise
    image = cv2.bilateralFilter(image, d=9, sigmaColor=70, sigmaSpace=150)
    # image = cv2.fastNlMeansDenoisingColored(image, None, h=10, hColor=10, templateWindowSize=11, searchWindowSize=21)

    # sharpen edges
    # image = cv2.GaussianBlur(image, (3, 3), 0)
    laplacian = cv2.Laplacian(image, cv2.CV_8U)
    # image = cv2.subtract(image, laplacian)

    # create mask using thresholding for missing circle region
    # mask = cv2.inRange(image, 30, 230)
    # mask = cv2.bitwise_not(mask)
    grey_mask = cv2.inRange(grey_img, 0, 10)
    colour_mask = cv2.inRange(image, (0, 0, 0), (10, 10, 10))

    # inpaint using circle mask
    # grey_img = cv2.inpaint(grey_img, grey_mask, 9, cv2.INPAINT_NS)
    # image = cv2.inpaint(image, colour_mask, 9, cv2.INPAINT_NS)
    # image = criminisi_inpaint(image, colour_mask)

    binary_mask = np.zeros(grey_img.shape, np.uint8)
    binary_mask[grey_mask > 0] = 1

    inpainter = Criminisi_Inpainter(image, binary_mask, patch_size=15, verbose=True, show_progress=False)
    # image = inpainter.inpaint()

    # band pass filter
    # image, magnitude_spectrum = band_pass(grey_img, 1, 20)

    # remove red 'R' from image
    # image = remove_R(image)

    # non local means filtering (https://docs.opencv.org/3.4/d5/d69/tutorial_py_non_local_means.html)
    # grey_img = cv2.fastNlMeansDenoising(grey_img, None, 10, 7, 21)
    # image = cv2.fastNlMeansDenoisingColored(image, None, h=8, hColor=10, templateWindowSize=7, searchWindowSize=21)
    # use greyscale nlm on different colour channels w different h values

    # add false colour
    colour_mapped = cv2.applyColorMap(grey_img, cv2.COLORMAP_TURBO) # turbo or jet?
    # image = colour_transfer(image, colour_mapped)
    # image = grey_img

    return image

# TESTING ======================================================================
isTesting = True

healthy = 'im001-healthy.jpg'
healthy_4 = 'im004-healthy.jpg'
healthy_12 = 'im012-healthy.jpg'
pneumonia = 'im053-pneumonia.jpg'
pneumonia_100 = 'im100-pneumonia.jpg'

img_name = healthy_4
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
