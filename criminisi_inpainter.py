import numpy as np
import time
import cv2
from tqdm import tqdm


class Criminisi_Inpainter():
    '''
    From https://github.com/igorcmoura/inpaint-object-remover/blob/master/inpainter/inpainter.py.
    Adapted to use opencv rather than scipy and skimage.

    Implements Criminisi (2004) 'Region Filling and Object Removal by Exemplar-Based Image Inpainting'.

    Requires binary mask of missing region, same size as image.
        Pixels values: 1 for missing, 0 for known region.
    '''

    def __init__(self, image, mask, patch_size=9, verbose=False, show_progress=False):
        self.image = image
        self.mask = mask.round()
        self.patch_size = patch_size
        self.verbose = verbose
        self.show_progress = show_progress

        # non-argument attributes
        self.iheight, self.iwidth = self.image.shape[:2]
        # The working image and working mask start as copies of the original
        # image and mask.
        self.working_image = np.copy(self.image)
        self.working_mask = np.copy(self.mask)
        self.front = np.zeros([self.iheight, self.iwidth])

        # The confidence is initially the inverse of the mask, that is, the
        # target region is 0 and source region is 1.
        self.confidence = (1.0 - self.mask)  # float type
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
        total = int(np.sum(self.mask))

        # loop until whole region is filled
        with tqdm(total=total, desc='Inpainting', unit='px') as pbar:
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
                    print('Time to find best: %.2f seconds' %
                          (time.time()-find_start_time))

                self._update_image(target_pixel, source_patch)

                # keep track of how much region left to fill
                remaining = int(np.sum(self.working_mask))
                filled_this_iter = total - remaining - (total - int(np.sum(self.mask)) - pbar.n)
                pbar.update(total - remaining - pbar.n)

                if self.verbose:
                    tqdm.write('%d of %d completed' % (total - remaining, total))

                # break and end if region filled
                if remaining == 0:
                    break

                c += 1

        if self.verbose:
            print('Inpainting took %.2f seconds to complete' %
                  (time.time() - start_time))
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
            patch = self._get_patch(point)  # patch centred at p

            # update confidence of pixel p as in Criminisi paper
            new_confidence[point[0], point[1]] = np.sum(np.sum(
                self.confidence[patch[0][0]:patch[0][1]+1, patch[1][0]:patch[1][1]+1])) / self.patch_size**2

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
        norm[norm == 0] = 1  # avoid div by zero

        # return unit vctr of normal
        unit_normal = normal/norm
        return unit_normal

    def _calc_gradient_matrix(self):
        height, width = self.working_image.shape[:2]

        # create grey image w/ mask region = 0 (black)
        grey_image = cv2.cvtColor(self.working_image, cv2.COLOR_BGR2GRAY)
        grey_image[self.working_mask > 0] = 0

        # get gradient of grey image
        gradient = np.nan_to_num(np.array(np.gradient(grey_image)))
        # get magnitude of gradient
        gradient_val = np.sqrt(gradient[0]**2 + gradient[1]**2)
        # init matrix to store max gradient
        max_gradient = np.zeros([height, width, 2])

        front_positions = np.argwhere(self.front > 0)  # get pixels on front

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
        target_patch = self._get_patch(
            target_pixel)  # get patch centred on target pixel
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