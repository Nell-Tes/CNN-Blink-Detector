import cv2
import dlib
import numpy as np
from keras.models import load_model
from imutils import face_utils

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
cnn = load_model('blinkModel.hdf5')

height, width = 26, 34
thresh = 3
counter = 0
total = 0

def detect(img, cascade=face_cascade, minimumFeatureSize=(20, 20)):
    if cascade.empty():
        raise Exception("Error loading Haar Cascade xml file.")
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=1, minSize=minimumFeatureSize)
    if len(rects) == 0:
        return []
    rects[:, 2:] += rects[:, :2]
    return rects

def cropEyes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detect(gray, minimumFeatureSize=(80, 80))

    eyes = []
    for rect in rects:
        x1, y1, x2, y2 = rect
        face = dlib.rectangle(int(x1), int(y1), int(x2), int(y2))
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[36:42]
        rightEye = shape[42:48]

        for eye in [leftEye, rightEye]:
            (ex, ey, ew, eh) = cv2.boundingRect(np.array([eye]))
            roi = gray[ey:ey+eh, ex:ex+ew]
            roi = cv2.resize(roi, (width, height))
            roi = np.expand_dims(roi, axis=2)
            roi = roi.astype('float32') / 255.0
            eyes.append(roi)
    return eyes

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    eyes = cropEyes(frame)
    if len(eyes) == 2:
        eyes_np = np.array(eyes)
        eyes_np = np.reshape(eyes_np, (2, height, width, 1))
        preds = cnn.predict(eyes_np)
        pred_avg = np.mean(preds)

        if pred_avg < 0.3:
            counter += 1
        else:
            if counter >= thresh:
                total += 1
                print("Blink detected!")
            counter = 0

    cv2.putText(frame, f"Blinks: {total}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Blink Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()