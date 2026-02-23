"""Shared image processing and classification utilities."""

import os
import cv2
import numpy as np

from src.criminisi_inpainter import Criminisi_Inpainter


def fix_perspective(image, corner_quality=0.01, border_width=4):
    """Fix perspective distortion of an X-ray image.

    Args:
        image: Input BGR image.
        corner_quality: Quality threshold for Shi-Tomasi corner detection.
        border_width: Border size (pixels) to preserve around the warped image.

    Returns:
        Perspective-corrected image.
    """
    height, width = image.shape[:2]

    mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Isolate image and missing region outlines
    mask = cv2.inRange(mask, 0, 10)

    # Laplacian edge detection
    mask = cv2.Laplacian(mask, cv2.CV_8U, ksize=7)

    # Extract contours (only extreme outer contours)
    contours, _ = cv2.findContours(mask, mode=cv2.RETR_EXTERNAL,
                                   method=cv2.CHAIN_APPROX_SIMPLE)

    # Create mask for ROI outline (square)
    square_mask = np.zeros((height, width), np.uint8)
    cv2.drawContours(square_mask, contours, -1, (255, 255, 255), 1)

    # Get 4 corners using Shi-Tomasi corner detection
    corners = cv2.goodFeaturesToTrack(square_mask, 4, corner_quality, 10)
    corners = corners.reshape(4, 2)

    # Desired 4 corners after warp
    map_to = [[border_width, border_width],
              [width - border_width, border_width],
              [border_width, height - border_width],
              [width - border_width, height - border_width]]
    map_from = [[0., 0.]] * 4  # to store 4 corners of ROI

    # Sort corners of ROI to align with corners of frame
    # such that map_from[i] is closest corner to map_to[i]
    for i in range(len(map_to)):
        min_dist = float('inf')
        for c in corners:
            dist = np.linalg.norm(c - map_to[i])
            if dist < min_dist:
                min_dist = dist
                map_from[i] = c

    # Shift perspective using perspective transform
    M = cv2.getPerspectiveTransform(
        np.array(map_from, np.float32), np.array(map_to, np.float32))
    return cv2.warpPerspective(image, M, (width, height))



def inpaint_image(image, patch_size=9, dilation_size=4):
    """Inpaint missing regions in X-ray image.

    Args:
        image: Input BGR image with missing regions (black pixels).
        patch_size: Patch size for Criminisi inpainting.
        dilation_size: Size of the square dilation kernel applied to the mask.

    Returns:
        Inpainted image.
    """
    # Greyscale
    grey_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create mask of missing region for inpainting
    inpainting_mask = cv2.inRange(grey_img, 0, 10)

    # Add 10-pixel-wide black edge to mask
    # (don't inpaint image's existing black border)
    border_width = 10
    inpainting_mask[:border_width, :] = 0
    inpainting_mask[-border_width:, :] = 0
    inpainting_mask[:, :border_width] = 0
    inpainting_mask[:, -border_width:] = 0

    # Dilate mask to expand missing region
    # (this overlaps region to be inpainted with known region,
    # so more seamless inpainting)
    dilation_kernel = np.ones((dilation_size, dilation_size), np.uint8)
    inpainting_mask = cv2.dilate(
        inpainting_mask, dilation_kernel, iterations=1)

    # Turn into binary mask
    inpainting_mask[inpainting_mask > 0] = 1

    inpainter = Criminisi_Inpainter(image, inpainting_mask, patch_size=patch_size)
    return inpainter.inpaint()


def filter_noise(image, median_ksize=3, nlm_h=7, apply_sharpening=True):
    """Remove noise from X-ray image.

    Args:
        image: Input BGR image.
        median_ksize: Kernel size for median blur (must be odd).
        nlm_h: Filter strength for the NLM luminance channel.
        apply_sharpening: Whether to apply Laplacian edge sharpening.

    Returns:
        Denoised image.
    """
    # Remove salt & pepper with median filter
    image = cv2.medianBlur(image, median_ksize)

    # NL means (converts to LAB for filtering)
    # h corresponds to filter strength in L channel
    # hColor corresponds to filter strength in A and B channels
    image = cv2.fastNlMeansDenoisingColored(
        image, None, h=nlm_h, hColor=1,
        templateWindowSize=9, searchWindowSize=31)

    if apply_sharpening:
        # Sharpen edges
        lap_kernel = np.array([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ])
        laplacian = cv2.filter2D(image, cv2.CV_8U, lap_kernel)
        image = cv2.subtract(image, laplacian)

    return image


def adjust_color_contrast(image, gamma=1.0, clahe_clip_limit=1.5, tile_grid_size=8):
    """Adjust color and contrast of X-ray image.

    Args:
        image: Input BGR image.
        gamma: Gamma correction value applied to the L channel.
        clahe_clip_limit: Clip limit for CLAHE histogram equalisation.
        tile_grid_size: Tile grid size for CLAHE.

    Returns:
        Color and contrast adjusted image.
    """
    # BGR --> LAB
    # lightness_ch = lightness, a = green-->red, b = blue-->yellow
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness_ch, a, b = cv2.split(lab_image)

    # Histogram equalization on lightness channel
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit,
                            tileGridSize=(tile_grid_size, tile_grid_size))
    lightness_ch = clahe.apply(lightness_ch)

    # Gamma correction
    lightness_ch = np.clip(
        np.power(lightness_ch / 255.0, gamma) * 255.0, 0, 255
    ).astype(np.uint8)

    # LAB to BGR
    lab_image = cv2.merge((lightness_ch, a, b))
    return cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)


def process_image(
    image,
    apply_perspective=True,
    corner_quality=0.01,
    border_width=4,
    apply_inpainting=True,
    patch_size=9,
    dilation_size=4,
    median_ksize=3,
    nlm_h=7,
    apply_sharpening=True,
    apply_colour_contrast=True,
    gamma=1.0,
    clahe_clip_limit=1.5,
    tile_grid_size=8,
):
    """Process X-ray image through up to 4 stages, each individually optional.

    Stages: perspective correction, inpainting, noise filtering,
    colour/contrast adjustment.

    Args:
        image: Input BGR image.
        apply_perspective: Whether to run perspective correction.
        corner_quality: Quality threshold for Shi-Tomasi corner detection.
        border_width: Border size (pixels) for perspective correction.
        apply_inpainting: Whether to run Criminisi inpainting.
        patch_size: Patch size for Criminisi inpainting.
        dilation_size: Dilation kernel size for inpainting mask.
        median_ksize: Kernel size for median blur (must be odd).
        nlm_h: NLM luminance filter strength.
        apply_sharpening: Whether to apply Laplacian edge sharpening.
        apply_colour_contrast: Whether to run the colour/contrast stage.
        gamma: Gamma correction value for contrast adjustment.
        clahe_clip_limit: CLAHE clip limit.
        tile_grid_size: CLAHE tile grid size.

    Returns:
        Processed image.
    """
    if apply_perspective:
        image = fix_perspective(image, corner_quality=corner_quality,
                                border_width=border_width)
    if apply_inpainting:
        image = inpaint_image(image, patch_size=patch_size,
                              dilation_size=dilation_size)
    image = filter_noise(image, median_ksize=median_ksize, nlm_h=nlm_h,
                         apply_sharpening=apply_sharpening)
    if apply_colour_contrast:
        image = adjust_color_contrast(image, gamma=gamma,
                                      clahe_clip_limit=clahe_clip_limit,
                                      tile_grid_size=tile_grid_size)
    return image


def classify_images(processed_dir, model_path):
    """Run the pre-trained classifier on processed images and return accuracy.

    Args:
        processed_dir: Directory containing processed .jpg images.
        model_path: Path to the ONNX classifier model.

    Returns:
        Accuracy as a float in [0, 1].
    """
    model = cv2.dnn.readNetFromONNX(model_path)

    # Ground truth: images 1-50 are healthy, 51-100 are pneumonia
    healthy_ids = [f'im{str(i).zfill(3)}' for i in range(1, 51)]
    pneumonia_ids = [f'im{str(i).zfill(3)}' for i in range(51, 101)]

    names = sorted([f for f in os.listdir(processed_dir) if f.endswith('.jpg')])
    if ".DS_Store" in names:
        names.remove(".DS_Store")

    correct = 0

    for filename in names:
        img = cv2.imread(os.path.join(processed_dir, filename))

        if img is not None:
            blob = cv2.dnn.blobFromImage(img, 1.0 / 255, (256, 256),
                                         (0, 0, 0), swapRB=True, crop=False)
            model.setInput(blob)
            output = model.forward()

            if output > 0.5:
                if filename.startswith(tuple(pneumonia_ids)):
                    correct += 1
            else:
                if filename.startswith(tuple(healthy_ids)):
                    correct += 1

    return correct / len(names) if names else 0.0
