import cv2
import os
import Localization
import Recognize
import csv
import pandas as pd
from difflib import SequenceMatcher
from datetime import datetime

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
    if True:
        tstart = datetime.now()
        recognized_plates = []
        recognition_frames = []
        recognition_timestamp = []
        frame = 0
        capture = cv2.VideoCapture(file_path)
        success, img = capture.read()
        if not capture.isOpened():
            print("Error opening video stream or file")
        
        current_segment1 = None
        last_detected1 = None
        last_category1 = None
        current_segment2 = None
        last_detected2 = None
        last_category2 = None
        while success:
            if frame % sample_frequency == 0:
                cropped = Localization.plate_detection(img, False)
                if cropped is not None:
                    characters1, category1 = Recognize.segment_and_recognize(cropped, False)
                    if characters1 is not None:
                        if current_segment1 is None:
                            current_segment1 = Segment(category1, frame)
                        detected1, detected_characters1, detection_frame1 = current_segment1.update_characters(
                            characters1,
                            category1,
                            last_category1,
                            last_detected1,
                            frame)
                        if detected1:
                            last_category1 = category1
                            last_detected1 = detected_characters1
                            print("".join(detected_characters1), detection_frame1)
                            current_segment1 = None
                            recognized_plates.append("".join(detected_characters1))
                            recognition_frames.append(detection_frame1)
                            recognition_timestamp.append((detection_frame1 - 1) / 12)
                
            success, img = capture.read()
            frame += 1
        
        capture.release()
        with open(save_path, "w") as f:
            df = pd.DataFrame({'License plate': recognized_plates, 'Frame no.': recognition_frames,
                               'Timestamp(seconds)': recognition_timestamp})
            result = df.to_csv(index=False)
            f.write(result)
        if not f.closed:
            f.close()

        # code to speed test

        tend = datetime.now()
        print (tend - tstart)
        os.system(
            'evaluation.py --file_path=' + str(save_path) + ' --ground_truth_path=groundTruth.csv')
    else:
        if False:
            for i in range(1, 36):
                name = 'TrainingSet/Categorie I/frames/video' + str(19) + 'frame' + str(i * 3) + '.png'
                print(name)
                img = cv2.imread(name)
                cropped, location = Localization.plate_detection(img, True)
                if cropped is not None:
                    characters, category = Recognize.segment_and_recognize(cropped, True)
                if characters is not None:
                    print(characters)
        else:
            frame = 1
            capture = cv2.VideoCapture('TrainingSet/Categorie III/Video47_2.avi')
            success, img = capture.read()
            if not capture.isOpened():
                print("Error opening video stream or file")
            current_segment = None
            last_detected = None
            last_category = None
            while success:
                if frame % sample_frequency == 0:
                    first, first_location, second, second_location, found_two = Localization.plate_detection(img, True)
                    if found_two:
                        print(found_two)
                    if first is not None:
                        characters, category = Recognize.segment_and_recognize(first, True)
                        if characters is not None:
                            if current_segment is None:
                                current_segment = Segment(category, frame)
                            detected, detected_characters, detection_frame = current_segment.update_characters(
                                characters,
                                category,
                                last_category,
                                last_detected,
                                frame)
                            if detected:
                                last_category = category
                                last_detected = detected_characters
                                print("".join(detected_characters))
                                current_segment = None
                success, img = capture.read()
                frame += 1
    
    capture.release()


def order(first, first_location, second, second_location):
    (x1, y1), (height1, width1), angle1 = first_location
    (x2, y2), (height2, width2), angle1 = second_location
    if width1 < 85 and height1 > 85:
        x1, y1 = y1, x1
    if width2 < 85 and height2 > 85:
        x2, y2 = y2, x2
    if x1 > x2:
        return second, first
    else:
        return first, second


class Segment:
    def __init__(self, category, frame):
        self.characters = [''] * 8
        self.detected_characters = ['a'] * 8
        
        self.times_detected = 0
        self.category = category
        self.last_Frame = frame
        self.wrong_hits = 0
        self.detected = False
    
    def update_characters(self, characters, category, last_category, last_characters, frame):
        if last_category is not None:
            if category != self.category or similar(characters, last_characters) < 0.5:
                if self.times_detected == 1:
                    save_char = self.characters
                    save_frame = self.last_Frame
                    self.__init__(category, frame)
                    return True, save_char, save_frame
                elif self.times_detected > 1:
                    if last_category == self.category and similar(characters, last_characters) > 0.5:
                        for i, char in enumerate(last_characters):
                            if self.characters[i] is not None:
                                self.characters[i] = self.characters[i] + char
                            else:
                                self.characters[i] = characters[i]
                            self.times_detected += 1
                            self.detected = self.__check_if_detected()
                            self.last_Frame = frame
                            if self.detected:
                                save_char = self.characters
                                save_frame = self.last_Frame
                                self.__init__(category, frame)
                                return True, save_char, save_frame
                            else:
                                return False, None, None
                    else:
                        self.__init__(category, frame)
        
        for i, char in enumerate(characters):
            if self.characters[i] is not None:
                self.characters[i] = self.characters[i] + char
            else:
                self.characters[i] = characters[i]
        self.times_detected += 1
        self.detected = self.__check_if_detected()
        self.last_Frame = frame
        if self.detected:
            return self.detected, self.detected_characters, self.last_Frame
        return self.detected, None, None
    
    def __get_max_occurring_char(self, chars):
        count = [0] * 256
        max_count = -1
        c = ''
        for i in chars:
            count[ord(i)] += 1;
        
        tie = False
        for i in chars:
            if max_count < count[ord(i)]:
                max_count = count[ord(i)]
                c = i
            elif max_count == count[ord(i)] and i != c:
                tie = True
        if tie:
            return None
        else:
            return c
    
    def __check_if_detected(self):
        if self.times_detected < 5:
            return False
        else:
            for i in range(8):
                if self.detected_characters[i] == 'a':
                    
                    char = self.__get_max_occurring_char(self.characters[i])
                    if char is None:
                        return False
                    else:
                        self.detected_characters[i] = char
        return True


def similar(a, b):
    if a is None or b is None:
        return 0
    return SequenceMatcher(None, a, b).ratio()
