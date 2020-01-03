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


def capture_frame_process(file_path, sample_frequency, save_path, debug):
    if not debug:
        video = 1
        directory = os.fsencode(file_path)

        for file in os.listdir(directory):
            filename = os.fsdecode(file)

            if filename.endswith(".avi") or filename.endswith(".mp4"):
                print(filename)
                frame = 1
                filename = file_path + "/" + filename
                capture = cv2.VideoCapture(filename)
                success, img = capture.read()
                # Information for calculating if the picture has changed
                current_segment = None

                if capture.isOpened() == False:
                    print("Error opening video stream or file")

                if capture.isOpened():

                    if success:
                        # Get the first segment of the first fame
                        first_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        current_segment = Segment(first_frame, 1)
                    # Read until video is completed
                    while success:
                        if current_segment.done:
                            first_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            current_segment = Segment(first_frame, frame)
                        if frame != current_segment.start_time:
                            # Get the next segment
                            # TODO find threshold for when to 'decide' that another car is shown
                            current_segment.update_segment(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
                            if current_segment.is_changed(3000):
                                current_segment.get_duration(frame)
                                print(current_segment.start_time, current_segment.end_time)
                                current_segment.done = True
                        if not current_segment.detected:
                            cropped = Localization.plate_detection(img, debug)
                            if cropped is not None:
                                characters = Recognize.segment_and_recognize(cropped, debug)
                                if characters is not None:
                                    if current_segment.update_characters(characters):
                                        print(characters)

                        success, img = capture.read()
                        frame += 1
                capture.release()
                video += 1
    else:
        # used for debugging
        # TODO not working: 2, 3, 7, 17, 26, 31(no detection)
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


class Segment:
    def __init__(self, first_frame, start_time):
        self.done = False
        self.current_segment = None
        self.update_segment(first_frame)
        self.detected = False
        self.prev_segment = None
        self.start_time = start_time
        self.characters = np.array((6,), dtype='str')
        self.detected_characters = np.full((6, 1), 'a', dtype='str')
        self.times_detected = 0
        self.end_time = None
        self.duration = None

    def update_segment(self, frame):
        offset_x = int(((frame.shape[1] - 200) / 2))
        offset_y = int((frame.shape[0] - 200) / 2)
        self.prev_segment = self.current_segment
        self.current_segment = frame[offset_y:+offset_y + 200, offset_x:offset_x + 200]
        return self.current_segment

    def is_changed(self, threshold):
        # print(Recognize.mse(self.prev_segment, self.current_segment))
        return Recognize.mse(self.prev_segment, self.current_segment) > threshold

    def update_characters(self, characters):
        for i, char in enumerate(characters):
            characters[i].join(char)
        self.times_detected += 1
        self.detected = self.__check_if_detected
        return self.detected

    def __check_if_detected(self):
        if self.times_detected < 3:
            return False
        else:
            for i in range(6):
                if self.detected_characters[i] == 'a':
                    char = self.__get_max_occuring_char(self.characters[i])
                    if char is None:
                        return False
                    else:
                        self.detected_characters[i] = char
        return True

    def __get_max_occuring_char(self, chars):
        count = [0] * 256
        max = -1
        c = ''
        t = ''
        for i in chars:
            count[ord(i)] += 1;

        tie = False
        for i in chars:
            if max < count[ord(i)]:
                max = count[ord(i)]
                c = i
            if max == count[ord(i)]:
                tie = True
                t = i
        if tie:
            return None
        else:
            return c

    def get_duration(self, end_time):
        self.end_time = end_time - self.start_time
        self.duration = self.end_time
        return self.duration
