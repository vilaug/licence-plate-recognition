import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

"""
In this file, you need to define plate_detection function.
To do:
	1. Localize the plates and crop the plates
	2. Adjust the cropped plate images
Inputs:(One)
	1. image: captured frame in CaptureFrame_Process.CaptureFrame_Process function
	type: Numpy array (imread by OpenCV package)
Outputs:(One)
	1. plate_imgs: cropped and adjusted plate images
	type: list, each element in 'plate_imgs' is the cropped image(Numpy array)
Hints:
	1. You may need to define other functions, such as crop and adjust function
	2. You may need to define two ways for localizing plates(yellow or other colors)
"""


def plate_detection(image, debug):
    return extract_plate(image, debug)


def extract_plate(image, debug):
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Apply Gaussian filter

    # Mask the licence plate

    img_hsv = cv2.GaussianBlur(img_hsv, (5, 5), 0)
    lower_yellow = np.array([5, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    mask = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    # Dilate the image a bit while eroding so the edge is clearer

    # Apply canny,
    if debug:
        cv2.imshow('Mask', mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # Fill mask holes
    mask = ndi.binary_fill_holes(mask)
    mask = np.where(mask, 0, 255)
    mask = mask.astype(np.uint8)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=10)

    canny = cv2.Canny(mask, 60, 120, 3)
    if debug:
        cv2.imshow('Mask Edges', canny)
        cv2.waitKey(0)
    # Find contours and sort them and save the largest one as the cropped image

    # Find contours from edges and sort them by area
    cnts, hierarchy = cv2.findContours(canny.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cnts = sorted(cnts, key=lambda cnt: cv2.contourArea(cnt, False), reverse=True)
    contour_img = image.copy()
    for cnt in cnts:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)

        if debug:
            cv2.drawContours(contour_img, [cnt], -1, (0, 255, 0), 3)
            cv2.imshow('image', contour_img)
            cv2.waitKey(0)

        (x, y), (height, width), angle = rect
        if height > 0 and width > 0:
            # Check that the contour is the right shape
            if (3 < width / height < 6 and 0.4 > height / width > 0.15) or (
                    3 < height / width < 6 and 0.4 > width / height > 0.15):
                rect_size = (int(height), int(width))
                rect_center = (int(x), int(y))
                # If it is rotated too much swap angle
                if angle < -45.:
                    angle += 90.0;
                    rect_size = (rect_size[1], rect_size[0])
                m = cv2.getRotationMatrix2D((x, y), angle, 1.0)
                # Rotate whole image and crop out the contour
                rotated = cv2.warpAffine(image, m, (image.shape[1], image.shape[0]), cv2.INTER_CUBIC)
                cropped = cv2.getRectSubPix(rotated, rect_size, rect_center)
                if debug:
                    cv2.imshow('cropped', cropped)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                return cropped
    return None
