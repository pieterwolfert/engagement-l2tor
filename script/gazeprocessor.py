import numpy as np
import cv2
import time
import sys
import csv
from PIL import Image
sys.path.insert(0, '/home/pieter/projects/engagement-l2tor/gaze-following')
from predict_gaze import Gaze

def pythonGaze(video, starttime, duration):
    angles = []
    model_def = '/home/pieter/projects/engagement-l2tor/data/model/deploy_demo.prototxt'
    model_weights = '/home/pieter/projects/engagement-l2tor/data/model/binary_w.caffemodel'
    gazemachine = Gaze(model_def, model_weights)
    sidefaceCascade=cv2.CascadeClassifier('/home/pieter/projects/engagement-l2tor/data/lbpcascade_profileface.xml')
    faceCascade = cv2.CascadeClassifier('/home/pieter/projects/engagement-l2tor/data/haarcascade_frontalface_alt.xml')
    eye_cascade = cv2.CascadeClassifier('/home/pieter/projects/engagement-l2tor/data/haarcascade_eye.xml')
    video = cv2.VideoCapture(video)
    video.set(cv2.CAP_PROP_POS_MSEC,starttime)
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
    for i in range(duration):
        ret, frame = video.read()
        frame[:,:,2] += 10
        x_frame = np.shape(frame)[1]
        y_frame = np.shape(frame)[0]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.1, 3)
        if len(faces) == 0:
            faces = sidefaceCascade.detectMultiScale(gray, 1.1, 3)
        if len(faces) == 0:
            faces = sidefaceCascade.detectMultiScale(np.fliplr(gray), 1.1, 3)
        # Draw a rectangle around the faces
        x_face = 0
        y_face = 0
        alpha = 0
        for (x, y, w, h) in faces:
            x_face = int((x + (x+w))/2)
            y_face = int((y+ (y+h))/2) - 30
            alpha = w / x_frame
        e = [x_face/x_frame, y_face/y_frame,]
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if e[0] != 0.0 and e[1] != 0.0:
            predictions = gazemachine.getGaze(image, e, alpha)
            cv2.circle(frame, (predictions[0], predictions[1]), 10, (0, 255,0), 2)
            cv2.line(frame, (x_face, y_face), (predictions[0], predictions[1]), (0, 255, 0), 2)
            #cv2.imshow('Eyeframe', gazemachine.getEyeImage())
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything is done, release the capture
    video.release()
    cv2.destroyAllWindows()
    #return angles

def main():
    #duration in number of frames (25 fps, 10 seconds of video)
    duration = 250
    clips = []
    with open("/home/pieter/data/l2tor_eng_set/start_times.csv", "r", newline='\n') as fp:
        wr = csv.reader(fp)
        for row in wr:
            clips.append(row)
    for i in clips[:1]:
        foldername = i[0][:4] + "/"
        clip = "/home/pieter/data/l2tor_eng_set/" + foldername + i[0]
        starttime = (int(i[2]) * 60000) + (int(i[3]) *1000)
        pythonGaze(clip, starttime, 250)


if __name__=="__main__":
    main()
