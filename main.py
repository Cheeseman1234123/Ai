import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load Haar Cascade for face detection
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Load pre-trained emotion recognition model
emotion_model = load_model('emotion_model.hdf5')

# Emotion labels (FER2013 standard)
emotion_labels = [
    "Angry", "Disgust", "Fear",
    "Happy", "Sad", "Surprise", "Neutral"
]

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_classifier.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Extract face ROI
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray.astype('float') / 255.0
        roi_gray = np.reshape(roi_gray, (1, 48, 48, 1))

        # Predict emotion
        prediction = emotion_model.predict(roi_gray, verbose=0)
        max_index = np.argmax(prediction[0])
        emotion = emotion_labels[max_index]

        # Display emotion label
        cv2.putText(frame, emotion,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA)

    cv2.imshow('Real-Time Face Emotion Recognition', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()