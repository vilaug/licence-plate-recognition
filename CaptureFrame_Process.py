import cv2
import os
import pandas as pd
import Localization
import Recognize
import csv
import numpy as np

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


def get_first_size(img):
    return min(250, img.shape[0]), min(250, img.shape[1])


def get_size(img, first_size):
    shape = img.shape
    if first_size[0] > shape[0] or first_size[1] > shape[1]:
        segment_size = (min(first_size[0], shape[0]), min(first_size[1], shape[1]))
        return True, segment_size
    else:
        return False, first_size


def get_segment(img, size):
    offset_x = int(((img.shape[1] - size[1]) / 2))
    offset_y = int((img.shape[0] - size[0]) / 2)
    return img[offset_y:size[0], offset_x:size[1]]


def capture_frame_process(file_path, sample_frequency, save_path, debug):
    if not debug:
        video = 1;
        directory = os.fsencode(file_path)

        for file in os.listdir(directory):
            filename = os.fsdecode(file)

            if filename.endswith(".avi"):
                print(filename)
                frame = 1
                times = []
                filename = file_path + "/" + filename
                capture = cv2.VideoCapture(filename)
                success, img = capture.read()
                # Information for calculating if the picture has shanged
                seg_start = 1
                first_frame = None
                first_segment = None
                first_size = None
                distances = []
                if success:
                    # Get the first segment of the first fame
                    first_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    first_size = get_first_size(first_frame)
                    first_segment = get_segment(first_frame, first_size)

                while success:
                    times.append(frame * sample_frequency)
                    if frame != 1:
                        # Get the next segment
                        # TODO add comparison with the start of the current segment
                        change_first, size = get_size(img, first_size)
                        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        segment = get_segment(img_gray, size)
                        if change_first:
                            first_segment = get_segment(first_frame, size)
                            first_size = size

                    if (frame - 1) % 6 == 0:
                        cropped = Localization.plate_detection(img, debug)
                        if cropped is not None:
                            # name = file_path + '/cropped/video' + str(video) + 'frame' + str(frame) + '.png'
                            # cv2.imwrite(name, cropped)
                            characters = Recognize.segment_and_recognize(cropped, debug)
                            if characters is not None:
                                print('Video ', video, ' frame: ', frame, ' at time: ', frame * sample_frequency, 's')
                                print(characters)
                    success, img = capture.read()
                    frame += 1

                # Write frame times
                text_file_path = file_path + '/frames/frame' + str(video) + '_times.txt'
                with open(text_file_path, 'w') as f:
                    f.write("[")
                    for item in times:
                        f.write("%s," % item)
                        f.write("]")
                video += 1
    else:
        # used for debugging
        #TODO save frames first
        for i in range(10):
            name = file_path + '/frames/video' + str(9) + 'frame' + str(1 + i * 6) + '.png'
            img = cv2.imread(name)
            cropped = Localization.plate_detection(img, debug)
            if cropped is not None:
                # cv2.imwrite(name, cropped)
                characters = Recognize.segment_and_recognize(cropped, debug)
                if characters is not None:
                    print(characters)
        # characters = Recognize.segment_and_recognize(cropped, False)
