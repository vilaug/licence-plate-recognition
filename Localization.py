import cv2
import numpy as np
import matplotlib.pyplot as plt

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


def plate_detection(image, write):
    return extract_plate(image, write)


def extract_plate(image, write):
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Apply Gaussian filter
    
    img_hsv = cv2.GaussianBlur(img_hsv, (5, 5), 0)
    # Mask the licence plate
    lower_yellow = np.array([10, 70, 105])
    upper_yellow = np.array([30, 255, 255])
    mask = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    # Dilate the image a bit while eroding so the edge is clearer
    kernel = np.ones((3, 3))
    
    # Apply canny, 1st threshold is 1/3 of the maximum
    if False:
        cv2.imshow('Mask', mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=10)
    canny = cv2.Canny(mask, 255 / 3, 255, 3)
    if False:
        cv2.imshow('Mask Edges', canny)
        cv2.waitKey(0)
    # Find contours and sort them and save the largest one as the cropped image
    cnts, hierarchy = cv2.findContours(canny.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=lambda cnt: cv2.arcLength(cnt, True), reverse=True)
    
    for cnt in cnts:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        if False:
            cv2.drawContours(image, [cnt], -1, (0,255,0), 3)
            cv2.imshow('image', image)
            cv2.waitKey(0)
        (x, y), (height, width), angle = rect
        if height > 0 and width > 0:
            if width / height > 2.5 and height / width < 0.4 or height / width > 2.5 and width / height < 0.4:
                box = np.int0(box)
                m, rotated, cropped = None, None, None
                rect_size = (int(height), int(width))
                rect_center = (int(x), int(y))
                if angle < -45.:
                    angle += 90.0;
                    rect_size = swap(rect_size[0], rect_size[1])
                m = cv2.getRotationMatrix2D((x, y), angle, 1.0)
                rotated = cv2.warpAffine(image, m, (image.shape[1], image.shape[0]), cv2.INTER_CUBIC)
                cropped = cv2.getRectSubPix(rotated, rect_size, rect_center)
                return cropped
    return ''


def swap(a, b):
    return b, a



## NENAUDOJAMA
def process_plate(image, write):
    # Convert to gray and equalize histogram
    image = cv2.GaussianBlur(image, (5, 5), 0)
    return image
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_gray = img_gray.astype('uint8')
    
    img_eq = cv2.equalizeHist(img_gray)
    blur = img_eq
    blur = cv2.GaussianBlur(blur, (3, 3), 0)
    
    return blur
    
    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
    th2 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 2)
    
    # plt.hist(img_eq.ravel(), 256)
    # plt.show()
    
    # Apply gaussian filter
    
    # Threshold
    
    # th3 = cv2.adaptiveThreshold(img_filter, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # ret, threshold = cv2.threshold(img_filter, find_iso_threshold(img_gray), 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((3, 3))
    
    # Morph, does not work
    img_morphed = th3
    img_morphed = cv2.dilate(img_morphed, kernel)
    img_morphed = cv2.erode(img_morphed, kernel)
    
    for x in range(1000):
        img_morphed = cv2.morphologyEx(img_morphed, cv2.MORPH_OPEN, kernel)
    
    if not True:
        cv2.imshow('Contrast enhanced with Gaussian and equalized', blur)
        cv2.waitKey(0)
        cv2.imshow('Equalized', img_eq)
        cv2.waitKey(0)
        cv2.imshow('Threshold with Median', th3)
        cv2.waitKey(0)
        
        cv2.imshow('Threshold with Gaussian', th2)
        cv2.waitKey(0)
        cv2.imshow('Morphed binary', img_morphed)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return th2
