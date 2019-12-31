import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
from skimage.filters import sobel
from skimage import morphology
from scipy import ndimage as ndi
from skimage.metrics import structural_similarity as ssim
from Characters import extractFromTTF
from sklearn import cluster
import string

"""
In this file, you will define your own segment_and_recognize function.
To do:
	1. Segment the plates character by character
	2. Compute the distances between character images and reference character images(in the folder of 'SameSizeLetters' and 'SameSizeNumbers')
	3. Recognize the character by comparing the distances
Inputs:(One)
	1. plate_imgs: cropped plate images by Localization.plate_detection function
	type: list, each element in 'plate_imgs' is the cropped image(Numpy array)
Outputs:(One)
	1. recognized_plates: recognized plate characters
	type: list, each element in recognized_plates is a list of string(Hints: the element may be None type)
Hints:
	You may need to define other functions.
"""


def segment_and_recognize(image, debug):
    return segment(image, debug)


def segment(image, debug):
    # Resize the image, convert to gray and equalize histogram
    image = cv2.resize(image, (int(image.shape[1] * (100 / image.shape[0])), 100))
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.equalizeHist(image_gray)

    # Apply bilateral filter
    blur1 = cv2.bilateralFilter(image_gray, 5, 150, 150)
    if debug:
        cv2.imshow('Contrast enhanced and equalized', blur1)
        cv2.waitKey()

    # Find elevation map from sobel edge detection
    elevation_map = sobel(blur1)

    # find 3 largest clusters and place markers there. 1st should always be the letters
    # Could be improved
    means = cluster.KMeans(n_clusters=3, max_iter=300, algorithm="elkan").fit(blur1.reshape(-1, 1))
    centers = means.cluster_centers_
    centers = sorted(centers)

    markers = np.zeros_like(blur1)
    markers[image_gray < centers[0]] = 2
    markers[image_gray > centers[1]] = 1
    segmentation = morphology.watershed(elevation_map, markers)

    if debug:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.imshow(segmentation, cmap=plt.cm.gray, interpolation='nearest')
        ax.axis('off')
        ax.set_title('segmentation')
        fig.show()
        np.set_printoptions(threshold=sys.maxsize)

    # Fill the holes
    segmentation_fill = ndi.binary_fill_holes(segmentation - 1)
    np.set_printoptions(threshold=sys.maxsize)
    if debug:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.imshow(segmentation_fill, cmap=plt.cm.gray, interpolation='nearest')
        ax.axis('off')
        ax.set_title('fill')
        fig.show()

    # Change to greyscale image
    segmentation_s = np.where(segmentation_fill, 0, 255)
    segmentation_s = segmentation_s.astype(np.uint8)
    if debug:
        cv2.imshow('seg', segmentation_s)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Erode
    segmentation_s = cv2.dilate(segmentation_s, np.ones((5, 5)), iterations=1)
    segmentation_s = cv2.erode(segmentation_s, np.ones((3, 3)), iterations=1)
    segmentation_s = cv2.dilate(segmentation_s, np.ones((3, 3)), iterations=1)
    segmentation_s = cv2.erode(segmentation_s, np.ones((5, 5)), iterations=1)

    # Get labeled image segments
    segmentation_fill = np.where(segmentation_s == 255, False, True)
    labeled_coins, n_objects = ndi.label(segmentation_fill)
    slices = ndi.find_objects(labeled_coins)
    slices = sorted(slices, key=lambda slice: slice[1].start)
    possible_chars = []
    for i in range(n_objects):
        curr_segment = segmentation[slices[i]]
        curr_segment = np.where(curr_segment > 1, 0, 255)
        curr_segment = curr_segment.astype(np.uint8)
        (height, width) = curr_segment.shape
        # print(width/height, height/width)
        if debug:
            cv2.imshow('segment', curr_segment)

            cv2.waitKey()
            cv2.destroyAllWindows()

        if width / height < 0.9 and height / width > 1.1 and height > image.shape[0] * 0.5:
            possible_chars.append(curr_segment)

    cv2.destroyAllWindows()
    if len(possible_chars) == 6:
        return get_characters(possible_chars, debug)

    return None


def get_characters(chars, debug):
    stock_chars, stock_numbers = extractFromTTF.get_stock_characters()
    recognized_chars = ""
    characters = string.digits + string.ascii_uppercase

    # TODO implement character/digit combination check

    # Go through each char and calculate mean square error, maybe switch to SSIM
    for (i, char) in enumerate(chars):
        errors = np.zeros((len(stock_chars), 2))
        for (j, stock_char) in enumerate(stock_chars):
            number = cv2.resize(stock_char, (char.shape[1], char.shape[0]))
            error = mse(char, number)
            errors[j] = [j, error]

        # Sort the errors
        errors = errors[errors[:, 1].argsort()]
        if debug:
            cv2.imshow('segment', char)
            cv2.imshow('stock', stock_chars[int(errors[0][0])])

            cv2.waitKey(0)
        recognized_chars += characters[int(errors[0][0])]
        if debug:
            recognized_chars += '(' + characters[int(errors[1][0])] + ')'
    return recognized_chars


def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    return err
