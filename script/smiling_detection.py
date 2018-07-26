import numpy as np
import cv2
import time

if __name__ == '__main__':
    faceCascade = cv2.CascadeClassifier('/home/pieter/projects/engagement-l2tor/data/haarcascade_frontalface_alt.xml')
    #eye_cascade = cv2.CascadeClassifier('/home/pieter/projects/engagement-l2tor/data/haarcascade_eye.xml')
    smileCascade = cv2.CascadeClassifier('/home/pieter/projects/engagement-l2tor/data/haarcascade_smile.xml')
    #video = cv2.VideoCapture('/home/pieter/Downloads/control_videos/406155_les6_fragment1.mp4')
    #video = cv2.VideoCapture('/dev/video0')
    video = cv2.VideoCapture('/mnt/disk1/documents/data/EmoReact_V_1.0/Data/Train/KIMCHI72_2.mp4')
    while True:
        # Capture frame-by-frame
        #time.sleep(1)
        ret, frame = video.read()
        x_frame = np.shape(frame)[1]
        y_frame = np.shape(frame)[0]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.3, 5, minSize=(55,55))
        # Draw a rectangle around the faces
        x_face = 0
        y_face = 0
        alpha = 0
        for (x, y, w, h) in faces:
            alpha = w / x_frame
            ind = int((y+h)/2)
            roi_color = frame[ind:y+h, x:x+w]
            smiles = smileCascade.detectMultiScale(roi_color, minSize=(40,40))
            for (ex, ey, ew, eh) in smiles:
                centre_face = (y + h)/2
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Display the resulting frame
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything is done, release the capture
    video.release()
    cv2.destroyAllWindows()
