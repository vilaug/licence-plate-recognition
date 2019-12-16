import cv2
import os
import pandas as pd
import Localization
import Recognize
import csv

"""
In this file, you will define your own CaptureFrame_Process funtion. In this function,
you need three arguments: file_path(str type, the video file), sample_frequency(second), save_path(final results saving path).
To do:
	1. Capture the frames for the whole video by your sample_frequency, record the frame number and timestamp(seconds).
	2. Localize and recognize the plates in the frame.(Hints: need to use 'Localization.plate_detection' and 'Recognize.segmetn_and_recognize' functions)
	3. If recognizing any plates, save them into a .csv file.(Hints: may need to use 'pandas' package)
Inputs:(three)
	1. file_path: video path
	2. sample_frequency: second
	3. save_path: final .csv file path
Output: None
"""


def capture_frame_process(file_path, sample_frequency, save_path):
    frames = 0
    capture = cv2.VideoCapture(file_path)
    count = 0
    times = []
    success, img = capture.read()

    # while success:
    #     cv2.imwrite("frames/frame%d.jpg" % count, img)  # save frame as JPEG file\
    #     print('Read a new frame: ', len(times), ' at time: ', frames*sample_frequency, 's')
    #     times.append(frames*sample_frequency)
    #     count += 1
    #     frames += 1
    #     success, img = capture.read()
    #
    # with open('frames/frame_times.txt', 'w') as f:
    #     f.write("[")
    #     for item in times:
    #             f.write("%s," % item)
    #     f.write("]")
    Localization.plate_detection(img)