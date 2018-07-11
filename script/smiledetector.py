import numpy as np
import cv2
import time
import sys
from PIL import Image

def smileDetect():
    sidefaceCascade=cv2.CascadeClassifier('/home/pieter/projects/engagement-l2tor/data/lbpcascade_profileface.xml')
    faceCascade = cv2.CascadeClassifier('/home/pieter/projects/engagement-l2tor/data/haarcascade_frontalface_alt.xml')
    smile_cascade = cv2.CascadeClassifier('/home/pieter/projects/engagement-l2tor/data/haarcascade_smile.xml')
    video = cv2.VideoCapture(0)
    while True:
        # Capture frame-by-frame
        #time.sleep(1)
        ret, frame = video.read()
        x_frame = np.shape(frame)[1]
        y_frame = np.shape(frame)[0]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.1, 3)
        # Draw a rectangle around the faces
        x_face = 0
        y_face = 0
        alpha = 0
        for (x, y, w, h) in faces:
            x_face = int((x + (x+w))/2)
            y_face = int((y+ (y+h))/2) - 30
            alpha = w / x_frame
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            smile = smile_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in smile:
                if ey > (.5*(y+h)):
                    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Display the resulting frame
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything is done, release the capture
    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    smileDetect()
