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
    if False:
        video = 0;
        directory = os.fsencode(file_path)
        
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(".avi"):
                    frame = 0
                    times = []
                    filename = file_path + "/"+filename
                    capture = cv2.VideoCapture(filename)
                    success, img = capture.read()
                    while success:
                        image_file_path = file_path+"/frames/video" + str(video) + "frame" + str(frame) + ".jpg"
                        cv2.imwrite(image_file_path, img)  # save frame as JPEG file\
                        print('Read a new frame: ', frame, ' at time: ', frame * sample_frequency, 's')
                        times.append(frame * sample_frequency)
                        frame += 1
                        Localization.plate_detection(img, file_path, True, video , frame)
                        #Recognize.segment_and_recognize(img, file_path, True, video, frame)
                        success, img = capture.read()
    
                    text_file_path = file_path + '/frames/frame' + str(video) + '_times.txt'
                    with open(text_file_path, 'w') as f:
                        f.write("[")
                        for item in times:
                            f.write("%s," % item)
                        f.write("]")
                    video+=1
    else:
        processed = Localization.plate_detection(0, file_path, False, 3, 5)
        Recognize.segment_and_recognize(processed, file_path, False, 2, 1)


