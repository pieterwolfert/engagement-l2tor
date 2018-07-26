import numpy as np
import cv2
import time
from face_recognition import face_locations
import keras
from keras.models import model_from_json
from skimage.measure import block_reduce
from skimage.transform import resize

def liveCam(model):
    video = cv2.VideoCapture('/dev/video0')
    while True:
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
            print("Working")
            probabilities = model.predict(head)[0]
            print("Results: {}".format(probabilities))
        # Display the resulting frame
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything is done, release the capture
    video.release()
    cv2.destroyAllWindows()

def main():
    model = model_from_json(open('models/model_smiling.json').read())
    model.load_weights('models/weights_smiling.h5')
    liveCam(model)

if __name__ == '__main__':
    main()
