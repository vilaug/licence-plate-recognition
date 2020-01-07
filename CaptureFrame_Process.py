import cv2
import os
import Localization
import Recognize
import csv
import numpy as np
import matplotlib.pyplot as plt

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
        recognized_plates = []
        recognition_frames = []
        recognition_timestamp = []
        frame = 1
        capture = cv2.VideoCapture(file_path)
        success, img = capture.read()
        # Information for calculating if the picture has changed
        current_segment = None
        if not capture.isOpened():
            print("Error opening video stream or file")
        
        current_segment = None
        flag = False
        cropped = None
        while success:
            if frame % sample_frequency == 0:
                if flag:
                    img2 = cropped
                    img1 = img

                    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                    
                    orb = cv2.create_ORB(100, 1.5)

                    # find the keypoints and descriptors with SIFT
                    kp1, des1 = orb.detectAndCompute(img1, None)
                    kp2, des2 = orb.detectAndCompute(img2, None)

                    # create BFMatcher object
                    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

                    # Match descriptors.
                    matches = bf.match(des1, des2)

                    # Sort them in the order of their distance.
                    matches = sorted(matches, key=lambda x: x.distance)

                    # Draw first 10 matches.
                    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], flags=2, outImg=img1)

                    plt.imshow(img3), plt.show()
                    
                            
                cropped = Localization.plate_detection(img, debug)
                if cropped is not None:
                    characters, category = Recognize.segment_and_recognize(cropped, debug)
                    flag = True
                    if characters is not None:
                        if current_segment is None:
                            current_segment = Segment(category)
                        print(''.join(characters))
                        detected, detected_characters = current_segment.update_characters(characters,
                                                                                          category)
                        
                        if detected:
                            recognition_frames.append(frame)
                            recognition_timestamp.append((frame - 1) * 1 / 12)
                            recognized_plate = "".join(detected_characters)
                            print(recognized_plate, frame, (frame - 1) * 1 / 12)
                            
                            recognized_plates.append(recognized_plate)
                else:
                    flag = False
            success, img = capture.read()
            frame += 1
        
        capture.release()
        with open(save_path, "w") as f:
            wr = csv.writer(f, delimiter="\n")
            for i in range(len(recognition_frames)):
                wr.writeRow("".join(recognized_plates[i]) + "," + str(recognition_frames[i]) + "," +
                            str(recognition_timestamp[i]))
        if not f.closed:
            f.close()
        os.system('evaluation.py --file_path=' + str(save_path) + ' --ground_truth_path=groundTruth.csv')
    
    if not debug:
        if False:
            video = 1
            directory = os.fsencode(file_path)
            
            recognized_plates = []
            recognition_frames = []
            recognition_timestamp = []
            for file in os.listdir(directory):
                filename = os.fsdecode(file)
                
                if filename.endswith(".avi") or filename.endswith(".mp4"):
                    print(video)
                    frame = 1
                    filename = file_path + "/" + filename
                    capture = cv2.VideoCapture(filename)
                    success, img = capture.read()
                    # Information for calculating if the picture has changed
                    current_segment = None
                    if capture.isOpened() == False:
                        print("Error opening video stream or file")
                    
                    if capture.isOpened():
                        
                        current_segment = None
                        while success:
                            if frame % samp == 0:
                                
                                cropped = Localization.plate_detection(img, debug)
                                if cropped is not None:
                                    characters, category = Recognize.segment_and_recognize(cropped, debug)
                                    if characters is not None:
                                        if current_segment is None:
                                            current_segment = Segment(category)
                                        detected, detected_characters = current_segment.update_characters(characters,
                                                                                                          category)
                                        
                                        if detected:
                                            recognized_plates.append("".join(detected_characters))
                                            
                                            recognition_frames.append(frame)
                                            recognition_timestamp.append(frame - 1 * 1 / 12)
                                            print("".join(detected_characters), frame, (frame - 1) * 1 / 12)
                            
                            success, img = capture.read()
                            frame += 1
                    capture.release()
                    video += 1
    
    elif False:
        # used for debugging
        # TODO not working: 2, 3, 7, 17, 26, 31(no detection)
        for i in range(1, 36):
            name = file_path + '/frames/video' + str(18) + 'frame' + str(i * 3) + '.png'
            img = cv2.imread(name)
            cropped = Localization.plate_detection(img, debug)
            if cropped is not None:
                # cv2.imwrite(name, cropped)
                characters = Recognize.segment_and_recognize(cropped, debug)
                if characters is not None:
                    print(characters)


class Segment:
    def __init__(self, category):
        self.characters = [''] * 8
        self.detected_characters = ['a'] * 8
        self.times_detected = 0
        self.category = category
        self.wrong_hits = 0
        self.detected = False
    
    def update_characters(self, characters, category):
        if self.category != category:
            if self.detected or self.wrong_hits > 2:
                self.__init__(category)
            else:
                self.wrong_hits += 1
        if not self.detected:
            for i, char in enumerate(characters):
                if self.characters[i] is not None:
                    self.characters[i] = self.characters[i] + char
                else:
                    self.characters[i] = characters[i]
            self.times_detected += 1
            self.detected = self.__check_if_detected()
            if self.detected:
                return self.detected, self.detected_characters
            return self.detected, None
        else:
            return False, None
    
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
        if self.times_detected < 3:
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
