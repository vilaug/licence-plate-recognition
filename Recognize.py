import cv2
import numpy as np
import os

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


def segment_and_recognize(image, file_path, write, video, frame):
    if write:
        characters = segment(image, write)
    else:
        segment(image, write)


def segment(image, write):
    image_edges = cv2.Canny(image, 255 / 3, 255)
    cv2.imshow('contour', image_edges)
    cv2.waitKey(0)

    cnts, hierarchy = cv2.findContours(image_edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea,  reverse=True)
    rcts = []
    for i in range(len(cnts)):
        x, y, w, h = cv2.boundingRect(cnts[i])
        if w/h < 0.7 and h/w > 1.5:
            if (x, y, w, h) not in rcts:
                rcts.append(cv2.boundingRect(cnts[i]))


    rcts = sorted(rcts, key=lambda tup: tup[0])

    for i in range(6):
        x, y, w, h = rcts[i]
        print(rcts[i])
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow('contour', image)
        cv2.waitKey(0)



    cv2.imshow('edges', image_edges)
    cv2.waitKey(0)
