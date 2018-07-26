import numpy as np
import cv2
import os
import time
from face_recognition import face_locations
import keras
from keras.models import model_from_json
from skimage.measure import block_reduce
from skimage.transform import resize

def liveCam(model, video):
    video = cv2.VideoCapture(video)
    smile = 0.0
    frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    for i in range(int(video.get(cv2.CAP_PROP_FRAME_COUNT))):
        # Capture frame-by-frame
        #time.sleep(1)
        ret, frame = video.read()
        x_frame = np.shape(frame)[1]
        y_frame = np.shape(frame)[0]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_locations(frame)
        # Draw a rectangle around the faces
        for (y, x, h, w) in faces:
            cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
            head = gray[y:h, w:x]
            head = resize(head, (32,32))
            head = head.astype(np.float)
            head = np.array([[head]])
            head = np.transpose(head, [0, 2, 3, 1])
            probabilities = model.predict(head)[0]
            if probabilities[1] > smile:
                smile = probabilities[1]
        # Display the resulting frame
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything is done, release the capture
    video.release()
    cv2.destroyAllWindows()
    return smile

def main():
    model = model_from_json(open('models/model_smiling.json').read())
    model.load_weights('models/weights_smiling.h5')
    for filename in os.listdir('data/l2torclips/small'):
        if filename.endswith(".mp4"):
            print(filename)
            try:
                smile_pred = liveCam(model,'data/l2torclips/small/' + filename)
                print(smile_pred)
                stat = "{}, {}\n".format(filename, smile_pred)
                with open("smiling.txt", "a") as gz:
                    gz.write(stat)
            except ValueError as err:
                continue

    liveCam(model)

if __name__ == '__main__':
    main()
