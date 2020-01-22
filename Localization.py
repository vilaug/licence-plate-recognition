import cv2
import numpy as np
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


def plate_detection(image):
    # Convert the color space to HSV
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Use gaussian blur on the HSV image
    img_hsv = cv2.GaussianBlur(img_hsv, (5, 5), 0)
    
    # Define the mask and mask the image
    lower_yellow = np.array([5, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    mask = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
        
    # Fill mask holes(holes in the arrays that are have a contour
    mask = ndi.binary_fill_holes(mask)
    mask = np.where(mask, 0, 255)
    mask = mask.astype(np.uint8)
    
    # Canny edge detection
    # Apply Gaussian filter to smooth the image in order to remove the noise
    # Find the intensity gradients of the image
    # Apply non-maximum suppression to get rid of spurious response to edge detection
    # Apply double threshold to determine potential edges
    # Suppress all the other edges that are weak and not connected to strong edges.
    canny = cv2.Canny(mask, 60, 120, 3)
    # Find contours and sort them and save the largest one as the cropped image
    
    # Find contours from edges and sort them by area
    cnts, hierarchy = cv2.findContours(canny.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=lambda cnt: cv2.contourArea(cnt, False), reverse=True)
    contour_img = image.copy()
    flag = False
    first = None
    for cnt in cnts:
        rect = cv2.minAreaRect(cnt)
        (x, y), (height, width), angle = rect
        if height > 0 and width > 0:
            # Check that the contour is the right shape
            if (width > 80 and (3 < width / height < 6 and 0.4 > height / width > 0.15)) or ((
                                                                                                     3 < height / width < 6 and 0.4 > width / height > 0.15) and height > 80):
                rect_size = (int(height), int(width))
                rect_center = (int(x), int(y))
                # If it is rotated too much swap axis
                if angle < -45.:
                    angle += 90.0
                    rect_size = (rect_size[1], rect_size[0])
                m = cv2.getRotationMatrix2D((x, y), angle, 1.0)
                # Rotate whole image and crop out the contour
                rotated = cv2.warpAffine(image, m, (image.shape[1], image.shape[0]), cv2.INTER_CUBIC)
                cropped = cv2.getRectSubPix(rotated, rect_size, rect_center)
                return cropped
    return None
