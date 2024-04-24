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

    map_to = [[0, 0], [width, 0], [0, height], [width, height]]  # desired 4 corners after warp
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
        self.image = image
        self.mask = mask.round()
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
        self.confidence = (1.0 - self.mask) # float type
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
            # get fill front
            self._find_front()

            if self.show_progress:
                name = 'Results/working_image_' + str(c) + '.png'
                cv2.imwrite(name, self.working_image)

            self._update_priorities()

            # target pixel == highest priority
            target_pixel = self._find_highest_priority_pixel()

            find_start_time = time.time()
            source_patch = self._find_source_patch(target_pixel)

            if self.verbose:
                print('Time to find best: %.2f seconds' % (time.time()-find_start_time))

            self._update_image(target_pixel, source_patch)

            # keep track of how much region left to fill
            remaining = np.sum(self.working_mask)
            total = np.sum(self.mask)
            if self.verbose:
                print('%d of %d completed' % (total-remaining, total))

            # break and end if region filled
            if remaining == 0:
                break
            
            c += 1

        if self.verbose:
            print('Inpainting took %.2f seconds to complete' % (time.time() - start_time))
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

    def _update_priorities(self):
        self._update_confidence()
        self._update_data()
        # update priority for every pixel, P = C * D.
        # (multiply confidence and data matrix by fill front to apply to all
        #     pixels at once --> much more efficient than for loop!)
        self.priority = self.confidence * self.data * self.front

    def _update_confidence(self):
        new_confidence = np.copy(self.confidence)

        # list pix on fill front
        front_positions = np.argwhere(self.front > 0)
        # iterate through them
        for point in front_positions:
            patch = self._get_patch(point) # patch centred at p

            # update confidence of pixel p as in Criminisi paper
            new_confidence[point[0], point[1]] = np.sum(np.sum(
                self.confidence[patch[0][0]:patch[0][1]+1,
                                patch[1][0]:patch[1][1]+1]
                )) / self.patch_size**2

        self.confidence = new_confidence

    def _update_data(self):
        normal = self._calc_normal_matrix()
        gradient = self._calc_gradient_matrix()

        normal_gradient = normal * gradient
        self.data = np.sqrt(
            normal_gradient[:, :, 0]**2 + normal_gradient[:, :, 1]**2
            ) + 0.001  # make sure data > 0

    def _calc_normal_matrix(self):
        # create x and y kernels for normal calculation
        x_kernel = np.array([[.25, 0, -.25],
                             [.5, 0, -.5],
                             [.25, 0, -.25]])

        y_kernel = np.array([[-.25, -.5, -.25],
                             [0, 0, 0],
                             [.25, .5, .25]])

        # convolve kernels with mask
        x_normal = cv2.filter2D(self.working_mask.astype(float), -1, x_kernel)
        y_normal = cv2.filter2D(self.working_mask.astype(float), -1, y_kernel)

        # depth stack x and y normal to get 2D normal matrix
        normal = np.dstack((x_normal, y_normal))

        height, width = normal.shape[:2]
        # get norm
        norm = np.sqrt(y_normal**2 + x_normal**2)
        # reshape norm to 3D
        norm = norm.reshape(height, width, 1).repeat(2, axis=2)
        norm[norm == 0] = 1 # avoid div by zero

        # return unit vctr of normal
        unit_normal = normal/norm
        return unit_normal

    def _calc_gradient_matrix(self):
        height, width = self.working_image.shape[:2]

        # create grey image w/ mask region = 0 (black)
        grey_image = cv2.cvtColor(self.working_image, cv2.COLOR_BGR2GRAY)
        grey_image[self.working_mask > 0] = 0

        gradient = np.nan_to_num(np.array(np.gradient(grey_image))) # get gradient of grey image
        gradient_val = np.sqrt(gradient[0]**2 + gradient[1]**2) # get magnitude of gradient
        max_gradient = np.zeros([height, width, 2]) # init matrix to store max gradient

        front_positions = np.argwhere(self.front > 0) # get pixels on front

        # for each pixel on front:
        for p in front_positions:
            # get patch centred on p
            patch = self._get_patch(p)

            # get y gradient and x gradient of patch
            patch_y_gradient = gradient[0][
                patch[0][0]:patch[0][1]+1,
                patch[1][0]:patch[1][1]+1
                ]
            patch_x_gradient = gradient[1][
                patch[0][0]:patch[0][1]+1,
                patch[1][0]:patch[1][1]+1
                ]
            
            # get gradient values in patch
            patch_gradient_val = gradient_val[
                patch[0][0]:patch[0][1]+1,
                patch[1][0]:patch[1][1]+1
                ]

            # get position of max gradient in patch
            patch_max_pos = np.unravel_index(
                patch_gradient_val.argmax(),
                patch_gradient_val.shape
            )

            # store max gradient for pixel p
            max_gradient[p[0], p[1], 0] = patch_y_gradient[patch_max_pos]
            max_gradient[p[0], p[1], 1] = patch_x_gradient[patch_max_pos]

        # return matrix of max gradient for all pix on fill front
        return max_gradient

    def _find_highest_priority_pixel(self):
        point = np.unravel_index(self.priority.argmax(), self.priority.shape)
        return point

    def _find_source_patch(self, target_pixel):
        '''
        Given a target pixel, find its patch & get the best source patch for inpainting it.
        '''
        target_patch = self._get_patch(target_pixel) # get patch centred on target pixel
        height, width = self.working_image.shape[:2]
        patch_height = (1+target_patch[0][1]-target_patch[0][0])
        patch_width = (1+target_patch[1][1]-target_patch[1][0])

        # init best match so far and difference (euclidean distance) from target patch
        best_match = None
        best_match_difference = 0

        # use lab space since being perceptually uniform means
        # euclidean distance more closely represents perceptual difference
        lab_image = cv2.cvtColor(self.working_image, cv2.COLOR_BGR2LAB)

        # check all patches in image
        for y in range(height - patch_height + 1):
            for x in range(width - patch_width + 1):
                source_patch = [
                    [y, y + patch_height-1],
                    [x, x + patch_width-1]
                ]
                # skip if source patch overlaps with the region still to be filled in
                if self._patch_data(self.working_mask, source_patch).sum() != 0:
                    continue

                # compare differences between target and source patches
                # using sum of squared distances between already filled-in pix
                difference = self._calc_patch_difference(
                    lab_image,
                    target_patch,
                    source_patch
                )

                # update best match if improved
                if best_match is None or difference < best_match_difference:
                    best_match = source_patch
                    best_match_difference = difference
        return best_match

    def _update_image(self, target_pixel, source_patch):
        '''
        Fill in 'empty' pix in target patch using source patch.
        '''
        target_patch = self._get_patch(target_pixel)

        # get pix in target patch to be filled in
        pixels_positions = np.argwhere(
            self._patch_data(
                self.working_mask,
                target_patch
            ) > 0
        ) + [target_patch[0][0], target_patch[1][0]]

        # set confidence of these unfilled pix to be same as target pixel
        patch_confidence = self.confidence[target_pixel[0], target_pixel[1]]
        for point in pixels_positions:
            self.confidence[point[0], point[1]] = patch_confidence

        # get section of mask to be filled in this iteration
        mask = self._patch_data(self.working_mask, target_patch)
        bgr_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # get source and target patches in actual image
        source_data = self._patch_data(self.working_image, source_patch)
        target_data = self._patch_data(self.working_image, target_patch)

        # new data from source patch where mask is 1 (to fill in),
        # target patch where mask is 0 (already filled)
        new_data = source_data*bgr_mask + target_data*(1-bgr_mask)

        # update working image and mask
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

        # get mask of target patch
        mask = 1 - self.working_mask[
            target_patch[0][0]:target_patch[0][1]+1,
            target_patch[1][0]:target_patch[1][1]+1
            ]
        bgr_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # get data from patches
        target_data = image[
            target_patch[0][0]:target_patch[0][1]+1,
            target_patch[1][0]:target_patch[1][1]+1
        ] * bgr_mask
        source_data = image[
            source_patch[0][0]:source_patch[0][1]+1,
            source_patch[1][0]:source_patch[1][1]+1
         ] * bgr_mask
        # sum of squared distances (SSD) between target and source patches
        ssd = np.sum(((target_data - source_data)**2))
        # tie-breaker factor is euclidean dist between patch centres
        euclidean_distance = np.sqrt(
            (target_patch[0][0] - source_patch[0][0])**2 +
            (target_patch[1][0] - source_patch[1][0])**2
        )
        return ssd + euclidean_distance

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

# Main function
def process_image(image):
    '''
    Process xray image. Keeps same resolution.
    '''
    # fix warped perspective
    image = fix_perspective(image)

    # create mask of missing region for inpainting
    grey_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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

    # turn into binary mask
    inpainting_mask[inpainting_mask > 0] = 1

    inpainter = Criminisi_Inpainter(image, inpainting_mask, patch_size=15, verbose=True, show_progress=False)
    image = inpainter.inpaint()
   
    return image

# TESTING ======================================================================
isTesting = True

healthy_1 = 'im001-healthy.jpg'
healthy_4 = 'im004-healthy.jpg' # example on gc
healthy_12 = 'im012-healthy.jpg'
healthy_31 = 'im031-healthy.jpg'
healthy_42 = 'im042-healthy.jpg' # bad criminisi
pneumonia_53 = 'im053-pneumonia.jpg'
pneumonia_56 = 'im056-pneumonia.jpg' # bad contrast
pneumonia_74 = 'im074-pneumonia.jpg' # hard to see red R
pneumonia_100 = 'im100-pneumonia.jpg'

img_name = healthy_31
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
        cv2.imwrite(tag, processed)
    else:
        print(tag + " failed to load.")
