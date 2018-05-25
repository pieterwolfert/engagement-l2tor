import numpy as np
import cv2
import time
from predict_gaze import Gaze

if __name__ == '__main__':
    model_def = '/home/pieter/projects/engagement-l2tor/data/model/deploy_demo.prototxt'
    model_weights = '/home/pieter/projects/engagement-l2tor/data/model/binary_w.caffemodel'
    gazemachine = Gaze(model_def, model_weights)
    faceCascade=cv2.CascadeClassifier('/home/pieter/projects/engagement-l2tor/data/haarcascade_frontalface_alt.xml')
    print(faceCascade.empty())
    video = cv2.VideoCapture(0)
    while True:
        # Capture frame-by-frame
        #time.sleep(1)
        ret, frame = video.read()
        x_frame = np.shape(frame)[1]
        y_frame = np.shape(frame)[0]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.3, 5)
        # Draw a rectangle around the faces
        x_face = 0
        y_face = 0
        for (x, y, w, h) in faces:
            x_face = int((x + (x+w))/2)
            y_face = int((y+ (y+h))/2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Display the resulting frame
        e = [x_face/x_frame, y_face/y_frame,]
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if e[0] != 0.0 and e[1] != 0.0:
            predictions = gazemachine.getGaze(image, e)
            cv2.imshow('Eyeframe', gazemachine.getEyeImage())
            #cv2.circle(frame, (predictions[0], predictions[1]), 10, (0, 255,0), 2)
            cv2.line(frame, (x_face, y_face), (predictions[0], predictions[1]), (0, 255, 0), 2)
            print(predictions)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything is done, release the capture
    video.release()
    cv2.destroyAllWindows()
