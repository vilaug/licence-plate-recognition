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


def plate_detection(image, file_path, write, frame, video):
    if write:
        cropped = extract_plate(image)
        if cropped != '':
            crop_file_path = file_path + "/cropped/video" + str(video) + "frame" + str(frame) + ".jpg"
            cv2.imwrite(crop_file_path, cropped)
            processed = process_plate(cropped, write)
            processed_file_path = file_path + "/processed/video" + str(video) + "frame" + str(frame) + ".jpg"
            cv2.imwrite(processed_file_path, processed)
    else:
        filename = file_path + "/frames/video" + str(video) + "frame" + str(frame) + ".jpg"
        img = cv2.imread(filename)
        cropped = extract_plate(img)
        process_plate(cropped, write)


def extract_plate(image):
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([15, 70, 110])
    upper_yellow = np.array([30, 255, 255])
    mask = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    cv2.bitwise_and(image, image, mask=mask)
    kernel = np.ones((3, 3))
    mask_ed = mask
    for x in range(2):
        mask_ed = cv2.erode(mask_ed, kernel, iterations=2)
        mask_ed = cv2.dilate(mask_ed, kernel, iterations=2)
    
    mask_edges = cv2.Canny(mask_ed, 200, 200)
    pts = np.argwhere(mask_edges > 0)
    if len(pts) > 0:
        y1, x1 = pts.min(axis=0)
        y2, x2 = pts.max(axis=0)
        cropped = image[y1:y2, x1:x2]
        return cropped
    return ''


def process_plate(image, write):
    image = image.astype('uint8')
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_eq = cv2.equalizeHist(img_gray)
    
    # plt.hist(img_eq.ravel(), 256)
    # plt.show()
    kernel_3x3 = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    kernel_3x3 = kernel_3x3 * (1 / kernel_3x3.sum())
    img_filter = cv2.filter2D(img_eq, -1, kernel_3x3)
    
    ret, threshold = cv2.threshold(img_filter, find_iso_threshold(img_gray), 255, cv2.THRESH_BINARY)
    if not write:
        cv2.imshow('Equalized', img_eq)
        cv2.waitKey(0)
        cv2.imshow('Contrast enhanced with Gaussian and equalized', img_filter)
        cv2.waitKey(0)
        cv2.imshow('Threshold with ISO data', threshold)
        cv2.waitKey(0)
    kernel = np.ones((3, 3))
    img_morphed = threshold
    for x in range(2):
        img_morphed = cv2.erode(img_morphed, kernel, iterations=1)
        img_morphed = cv2.dilate(img_morphed, kernel, iterations=1)
    
    return threshold


def find_iso_threshold(img):
    e = 0.2
    hist, bins = np.histogram(img.ravel(), 256, [0, 256])
    t = (1 + 255) / 2
    if img.shape[0] > 0 and img.shape[1] > 0:
        
        gmin = np.amin(img)
        gmax = np.amax(img)
        i = gmin + 1
        
        while True:
            nominator = 0
            denominator = 0
            for j in range(gmin, i):
                nominator += j * hist[j]
                denominator += hist[j]
            if denominator != 0:
                m1 = nominator / denominator
            else:
                m1 = 0
            nominator = 0
            denominator = 0
            for j in range(i, gmax + 1):
                nominator += j * hist[j]
                denominator += hist[j]
            if denominator != 0:
                m2 = nominator / denominator
            else:
                m2 = 0
            
            new_t = (m1 + m2) / 2
            i += 1
            
            difference = abs(t - new_t)
            if difference > e:
                t = new_t
            else:
                new_t = t
                break
            if i == gmax + 1:
                break
    return t
