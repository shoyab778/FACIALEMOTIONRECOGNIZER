import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load the model
try:
    model = keras.models.load_model(r"C:\face_emotion\images\facialemotionmodel.h5")
  # Change path if needed
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Open webcam
cap = cv2.VideoCapture(0)  # 0 for default webcam

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to capture image")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]  # Extract face
        roi_gray = cv2.resize(roi_gray, (48, 48))  # Resize to match model input
        roi_gray = np.expand_dims(roi_gray, axis=-1)  # Add channel dimension
        roi_gray = np.expand_dims(roi_gray, axis=0)  # Add batch dimension
        roi_gray = roi_gray / 255.0  # Normalize

        # Predict emotion
        prediction = model.predict(roi_gray)
        emotion = emotion_labels[np.argmax(prediction)]

        # Draw rectangle around face and label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Show video feed
    cv2.imshow("Emotion Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
