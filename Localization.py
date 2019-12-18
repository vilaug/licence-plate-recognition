import cv2
import numpy as np

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

	for i in range(2091):
		img = cv2.imread('frames/frame%d.jpg' % i)
		img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

		lower_yellow = np.array([15, 70, 110])
		upper_yellow = np.array([30, 255, 255])
		mask = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
		res = cv2.bitwise_and(img, img, mask=mask)

		kernel = np.ones((3,3))
		mask_ed = mask

		for x in range(2):
			mask_ed = cv2.erode(mask_ed, kernel, iterations=2)
			mask_ed = cv2.dilate(mask_ed, kernel, iterations=2)



		mask_edges = cv2.Canny(mask_ed, 200, 200)
		pts = np.argwhere(mask_edges > 0);
		if(len(pts)>0):
			y1, x1 = pts.min(axis=0)
			y2, x2 = pts.max(axis=0)
			cropped = img[y1:y2, x1:x2]
			print(i)
			cv2.imwrite('cropped/frame%d.jpg' % i, cropped)  # save frame as JPEG file\
