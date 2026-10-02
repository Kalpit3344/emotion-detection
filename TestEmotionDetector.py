import cv2
import numpy as np
from keras.models import model_from_json

emotion_dict = {0:"Angry", 1:"Disgusted", 2:"Fearful",
                3:"Happy", 4:"Neutral", 5:"Sad", 6:"Surprised"}

# Load model
with open('model/emotion_model.json', 'r') as f:
    loaded_model_json = f.read()
emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights("model/emotion_model.h5")

cap = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48,48))
        roi = np.expand_dims(np.expand_dims(roi, -1), 0) / 255.0
        prediction = emotion_model.predict(roi)
        label = emotion_dict[int(np.argmax(prediction))]
        cv2.putText(frame, label, (x+5, y-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()