import numpy as np
import math
import cv2
import time
import sys
import os
import csv
import face_recognition
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip, ffmpeg_resize
from skvideo.io.ffmpeg import FFmpegReader

from PIL import Image
sys.path.insert(0, '/home/pieter/projects/engagement-l2tor/gaze-following')
from predict_gaze import Gaze

def extractClips():
    #duration in number of frames (25 fps, 10 seconds of video)
    duration = 10
    clips = []
    with open("/mnt/disk2/l2tor_eng_set/start_times.csv", "r", newline='\n') as fp:
        wr = csv.reader(fp)
        for row in wr:
            clips.append(row)
    for i in clips:
        print("Processing clip {} of {}".format(i, len(clips)))
        foldername = i[0][:4] + "/"
        clip = "/mnt/disk2/l2tor_eng_set/" + foldername + i[0]
        starttime = (int(i[2]) * 60) + int(i[3])
        ffmpeg_extract_subclip(clip, starttime, starttime+duration, targetname="data/l2torclips/" + i[0][:-4] + "_" + i[1] + ".MTS")

def extractGaze(gazemachine, video):
    robot = 0
    tablet = 0
    other = 0
    video = cv2.VideoCapture(video)
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
    frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    print("Starting Now...")
    for i in range(int(video.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = video.read()
        x_frame = np.shape(frame)[1]
        y_frame = np.shape(frame)[0]
        e = [0.0, 0.0]
        faces = face_recognition.face_locations(frame)
        for (y, x, h, w) in faces:
            x_face = int(w + ((x - w)/2))
            y_face = int(h + ((y - h)/2) - 20)
            alpha = w / x_frame
            e = [x_face/x_frame, y_face/y_frame,]
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if e[0] != 0.0 and e[1] != 0.0:
            print("Head Detected")
            predictions = gazemachine.getGaze(image, e, alpha)
            cv2.circle(frame, (predictions[1], predictions[0]), 10, (0, 255,0), 2)
            cv2.line(frame, (x_face, y_face), (predictions[1], predictions[0]), (0, 255, 0), 2)
            cv2.circle(frame, (int(.20*x_frame), int(.50*y_frame)), 10, (0, 100,0), 2)
            try:
                degrees = math.degrees(math.atan((y_face - predictions[0])/(x_face - predictions[1])))
            except ZeroDivisionError as err:
                degrees = 0
            if degrees < 0 and degrees > -45:
                robot += 1
            elif degrees < -45 and degrees > -180:
                tablet += 1
            elif degrees > 0 and degrees < 180:
                other += 1
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()
    return frames, robot, tablet, other

def main():
    model_def = '/home/pieter/projects/engagement-l2tor/data/model/deploy_demo.prototxt'
    model_weights = '/home/pieter/projects/engagement-l2tor/data/model/binary_w.caffemodel'
    gazemachine = Gaze(model_def, model_weights)
    for filename in os.listdir('data/l2torclips/small'):
        if filename.endswith(".mp4"):
            print(filename)
            try:
                frames, robot, tablet, other = extractGaze(gazemachine, 'data/l2torclips/small/' + filename)
                stat = "{}, {}, {}, {}, {} \n".format(filename, frames, robot, tablet, other)
                with open("gaze2stats.txt", "a") as gz:
                    gz.write(stat)
            except ValueError as err:
                continue

if __name__=="__main__":
    main()
