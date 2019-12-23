import cv2
import numpy as np
import os
import skimage.filters as sk

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


## Write - boolean ar rasyt ar testuojant daryt kazka
def segment_and_recognize(image, write):
    if write:
        characters = segment(image, write)
    else:
        segment(image, write)


def segment(image, write):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.equalizeHist(image_gray)
    #blur = cv2.GaussianBlur(image_gray, (5, 5), 0)
    isodata_threshold = sk.threshold_isodata(image_gray)
    print(isodata_threshold)
    ret3, th3 = cv2.threshold(image_gray, isodata_threshold, 255, cv2.THRESH_BINARY)

    image_edges = cv2.Canny(th3, 255 / 3, 255)
    cv2.imshow('contour', th3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    cnts, hierarchy = cv2.findContours(image_edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=lambda cnt: cv2.arcLength(cnt, True), reverse=True)
    rcts = []
    for i in range(len(cnts)):
        x, y, w, h = cv2.boundingRect(cnts[i])
        print( x, y, w, h)
        if w / h < 0.73 and h / w > 1.4:
            if (x, y, w, h) not in rcts:
                flag = True
                
                for rct in rcts:
                    x1, y1, w1, h1 = rct
                    if (x1 >= x and x1 <= x + w) or (x1 <= x and x1 + w1 >= x):
                        flag = False
                
                if flag:
                    print('added')
                    rcts.append(cv2.boundingRect(cnts[i]))

        cv2.drawContours(image, [cnts[i]], -1, (0, 255, 0), 3)
        cv2.imshow('contour', image)
        cv2.waitKey(0)
    
    rcts = sorted(rcts, key=lambda tup: tup[0])
    
    for i in range(6):
        x, y, w, h = rcts[i]
        print(rcts[i])
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 50 * i, 0), 2)
        cv2.imshow('contour', image)
        cv2.waitKey(0)
    
    cv2.imshow('edges', image_edges)
    cv2.waitKey(0)
