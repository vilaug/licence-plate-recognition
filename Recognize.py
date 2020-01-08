import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
from skimage.filters import sobel
from skimage import morphology
from scipy import ndimage as ndi
from Characters import extractFromTTF
from sklearn import cluster
import string
import time

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
    blur1 = cv2.GaussianBlur(image_gray, (5, 5), cv2.BORDER_DEFAULT)
    if debug:
        cv2.imshow('Contrast enhanced and equalized', blur1)
        cv2.waitKey()
    
    # Find elevation map from sobel edge detection
    elevation_map = sobel(blur1)
    
    # Place the markers in the interest points
    # less than 35 is black and should be the characters
    # more than 80 should be the background
    markers = np.zeros_like(blur1)
    markers[image_gray < 35] = 2
    markers[image_gray > 80] = 1
    
    # Segments the characters using morphology watershed operation
    # Which behaves as a water shed in real life
    # Takes the elevation map from sobel edge detections and markers
    # For what to consider objects and what to consider background.
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
    if debug:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.imshow(segmentation_fill, cmap=plt.cm.gray, interpolation='nearest')
        ax.axis('off')
        ax.set_title('fill')
        fig.show()
    
    # Change to greyscale image, since it is now a boolean array containing
    # True where an object was detected
    # False where background was found
    segmentation_s = np.where(segmentation_fill, 0, 255)
    segmentation_s = segmentation_s.astype(np.uint8)
    if debug:
        cv2.imshow('seg', segmentation_s)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Make the characters thinner
    segmentation_s = cv2.dilate(segmentation_s, np.ones((5, 5)), iterations=1)
    segmentation_s = cv2.erode(segmentation_s, np.ones((3, 3)), iterations=1)
    segmentation_s = cv2.dilate(segmentation_s, np.ones((3, 3)), iterations=1)
    segmentation_s = cv2.erode(segmentation_s, np.ones((5, 5)), iterations=1)
    
    # Get labeled image segments
    segmentation_fill = np.where(segmentation_s == 255, False, True)
    
    # Label the coins based on regions where a 'whole' object was detected
    labeled_coins, n_objects = ndi.label(segmentation_fill)
    
    # Sort the objects by x coordinate
    slices = ndi.find_objects(labeled_coins)
    slices = sorted(slices, key=lambda slice: slice[1].start)
    possible_chars = []
    last_end = 0
    
    index = 0
    ends = []
    for i in range(n_objects):
        curr_segment = segmentation[slices[i]]
        curr_segment = np.where(curr_segment > 1, 255, 0)
        curr_segment = curr_segment.astype(np.uint8)
        (height, width) = curr_segment.shape
        # print(width/height, height/width)
        if width / height < 0.9 and height / width > 1.1 and height > image.shape[0] * 0.5:
            if last_end == 0:
                last_end = slices[i][1].stop
            elif last_end + 20 < slices[i][1].start:
                ends.append(index)
                last_end = slices[i][1].stop
                possible_chars.append(None)
            else:
                last_end = slices[i][1].stop
            possible_chars.append(curr_segment)
            if False:
                cv2.imshow('segment', curr_segment)
                
                cv2.waitKey()
                cv2.destroyAllWindows()
            index += 1
    
    cv2.destroyAllWindows()
    # CATEGORY 0 -  x - xxx - xx
    # CATEGORY 1 - xx - xxx - x
    # CATEGORY 2 - xx - xx - xx
    # CATEGORY 3 - xxx - xx - x
    if len(possible_chars) == 8 and len(ends) == 2:
        if ends[0] == 1 and ends[1] == 4:
            return get_characters(possible_chars, debug, 0)
        elif ends[0] == 2:
            
            if ends[1] == 5:
                return get_characters(possible_chars, debug, 1)
            elif ends[1] == 4:
                return get_characters(possible_chars, debug, 2)
        elif ends[0] == 3 and ends[1] == 5:
            return get_characters(possible_chars, debug, 3)
    
    return None, None


def find_best_digit(recognized_chars, errors):
    min_error = sys.maxsize
    min_char = -1
    min_index = -1
    for i, char in enumerate(recognized_chars):
        if char != '-' and char not in string.ascii_uppercase:
            if errors[i][0][1] < min_error:
                min_error = errors[i][0][1]
                min_char = char
                min_index = i
    return min_index, min_char
    
    # CATEGORY 0    x - xxx - xx
    # CATEGORY 0.1  9 - XXX - 99
    # CATEGORY 0.2  X - 999 - XX
    # CATEGORY 1    xx - xxx - x
    # CATEGORY 1.1  99 - XXX - 9
    # CATEGORY 1.2  XX - 999 - X
    # CATEGORY 2    xx - xx - xx
    # CATEGORY 2.1  99 - XXX - X
    # CATEGORY 2.2  XX - 999 - X
    # CATEGORY 2.3  XX - XXX - 9
    # CATEGORY 3    xxx - xx - x
    # CATEGORY 3.1  XXX - 99 - X


def is_digits(best_index, best_digit, category):
    is_digits_arr = None
    new_category = category
    if category == 0:
        if best_index == 0 or best_index > 5:
            is_digits_arr = np.array([True, False, True])
            new_category = 0.1
        else:
            is_digits_arr = np.array([False, True, False])
            new_category = 0.2
    elif category == 1:
        if best_index < 2 or best_index == 7:
            is_digits_arr = np.array([True, False, True])
            new_category = 1.1
        else:
            is_digits_arr = np.array([False, True, False])
            new_category = 1.2
    elif category == 2:
        if best_index < 2:
            is_digits_arr = np.array([True, False, False])
            new_category = 2.1
        elif 2 < best_index < 5:
            is_digits_arr = np.array([False, True, False])
            new_category = 2.2
        else:
            is_digits_arr = np.array([False, False, True])
            new_category = 2.3
    elif category == 3:
        if 3 < best_index < 6:
            is_digits_arr = np.array([False, True, False])
            new_category = 3.1
    return is_digits_arr, new_category
    
    # CATEGORY 0    x - xxx - xx
    # CATEGORY 1    xx - xxx - x
    # CATEGORY 2    xx - xx - xx
    # CATEGORY 3    xxx - xx - x


def get_category_slices(category):
    if category == 0:
        return [(0, 1), (2, 5), (6, 8)]
    elif category == 1:
        return [(0, 2), (3, 6), (7, 8)]
    elif category == 2:
        
        return [(0, 2), (3, 5), (6, 8)]
    elif category == 3:
        return [(0, 3), (4, 6), (7, 8)]


def get_characters(chars, debug, category):
    characters, recognized_chars, errors = recognize(chars, debug)
    best_index, best_digit, = find_best_digit(recognized_chars, errors)
    slices = get_category_slices(category)
    is_digits_arr, category = is_digits(best_index, best_digit, category)
    
    # CATEGORY 0    x - xxx - xx
    # CATEGORY 1    xx - xxx - x
    # CATEGORY 2    xx - xx - xx
    # CATEGORY 3    xxx - xx - x
    j = 0
    if is_digits_arr is not None:
        for i, is_digit in enumerate(is_digits_arr):
            if is_digit:
                for j in range(slices[i][0], slices[i][1]):
                    count = 0
                    char_index = int(errors[j][count][0])
                    while characters[char_index] not in string.digits:
                        count += 1
                        char_index = int(errors[j][count][0])
                    if count != 0:
                        recognized_chars[j] = characters[char_index]
            else:
                for j in range(slices[i][0], slices[i][1]):
                    count = 0
                    char_index = int(errors[j][count][0])
                    while characters[char_index] in string.digits:
                        count += 1
                        char_index = int(errors[j][count][0])
                    if count != 0:
                        recognized_chars[j] = characters[char_index]
        return recognized_chars, category
    else:
        return None, None


def recognize(chars, debug):
    stock_chars, stock_numbers = extractFromTTF.get_stock_characters()
    recognized_chars = []
    characters = string.digits + string.ascii_uppercase
    # TODO implement character/digit combination check
    last_start = 0
    errors = np.zeros((8, len(stock_chars), 2))
    # Go through each char and calculate mean square error, maybe switch to SSIM
    for (i, char) in enumerate(chars):
        if char is None:
            recognized_chars.append("-")
        else:
            for (j, stock_char) in enumerate(stock_chars):
                stock_char = cv2.resize(stock_char, (char.shape[1], char.shape[0]))
                
                stock_char = np.where(stock_char > 0, 0, 255)
                error = mse(stock_char, char)
                errors[i][j] = [j, error]
            
            # Sort the errors
            
            errors[i] = errors[i][errors[i][:, 1].argsort()]
            if False:
                cv2.imshow('segment', char)
                cv2.imshow('stock', stock_chars[int(errors[i][0][0])])
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            recognized_chars.append(characters[int(errors[i][0][0])])
            if False:
                recognized_chars.append('(' + characters[int(errors[i][1][0])] + ')')
    return characters, recognized_chars, errors


def mse(image_a, image_b):
    err = np.sum((image_a.astype("float") - image_b.astype("float")) ** 2)
    err /= float(image_a.shape[0] * image_b.shape[1])
    return err
