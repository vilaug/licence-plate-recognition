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


def plate_detection(image, file_path, write, video, frame):
    if write:
        cropped = extract_plate(image, write)
        crop_file_path = file_path + "/cropped/video" + str(video) + "frame" + str(frame) + ".jpg"
        cv2.imwrite(crop_file_path, cropped)
        processed = process_plate(cropped, write)
        processed_file_path = file_path + "/processed/video" + str(video) + "frame" + str(frame) + ".jpg"
        cv2.imwrite(processed_file_path, processed)
    else:
        filename = file_path + "/frames/video" + str(video) + "frame" + str(frame) + ".jpg"
        img = cv2.imread(filename)
        cropped = extract_plate(img, write)
        process_plate(cropped, write)


def extract_plate(image, write):
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Apply Gaussian filter
    
    img_hsv = cv2.GaussianBlur(img_hsv, (5, 5), 0)
    
    # Mask the licence plate
    lower_yellow = np.array([9, 70, 90])
    upper_yellow = np.array([30, 255, 255])
    mask = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    cv2.bitwise_and(image, image, mask=mask)
    
    # Dilate the image a bit while eroding so the edge is clearer
    kernel = np.ones((3, 3))
    
    mask_ed = mask
    for x in range(2):
        mask_ed = cv2.dilate(mask_ed, kernel, iterations=3)
        mask_ed = cv2.erode(mask_ed, kernel, iterations=2)
    
    # Apply canny, 1st threshold is 1/3 of the maximum
    
    canny = cv2.Canny(mask_ed, 255 / 3, 255, 3)
    
    # Find contours and sort them and save the largest one as the cropped image
    cnts, hierarchy = cv2.findContours(canny.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    rect = cv2.minAreaRect(cnts[0])
    box = cv2.boxPoints(rect)
    (x, y), (width, height), angle = rect
    
    box = np.int0(box)
    m, rotated, cropped = None, None, None
    
    rect_size = (int(width), int(height))
    rect_center = (int(x), int(y))
    if angle < -45.:
        angle += 90.0;
        rect_size = swap(rect_size[0], rect_size[1])
    m = cv2.getRotationMatrix2D((x, y), angle, 1.0)
    rotated = cv2.warpAffine(image, m, (image.shape[1], image.shape[0]), cv2.INTER_CUBIC)
    cropped = cv2.getRectSubPix(rotated, rect_size, rect_center)
    if not write:
        cv2.imshow('Contours', cropped)
        cv2.waitKey(0)
    return cropped


def swap(a, b):
    return b, a


def process_plate(image, write):
    # Convert to gray and equalize histogram
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_gray = img_gray.astype('uint8')
    

    img_eq = cv2.equalizeHist(img_gray)
    blur = img_eq
    blur = cv2.GaussianBlur(blur, (5, 5), 0)
    

    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,9, 2)
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
    
    if not write:
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
        
    return img_morphed
