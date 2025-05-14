import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from datetime import datetime
import os
import tensorflow as tf

# Optional: Disable GPU errors if not using GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Emotion labels
emotion_dict = {
    0: "Angry", 1: "Disgusted", 2: "Fearful",
    3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"
}

# Load pre-trained emotion model
model_path = "/home/nandkumar/Desktop/Research Projet/emotion_model .h5"
emotion_model = load_model(model_path)
print("✅ Loaded emotion model from:", model_path)

# Initialize face detector and video capture
face_detector = cv2.CascadeClassifier(
    "/home/nandkumar/Desktop/Research Projet/haarcascade_frontalface_default.xml"
)
cap = cv2.VideoCapture(0)

# Emotion log list
emotion_log = []

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Could not access camera.")
            break

        frame = cv2.resize(frame, (1280, 720))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi_resized = cv2.resize(roi, (48, 48))
            roi_normalized = roi_resized / 255.0
            roi_input = np.expand_dims(np.expand_dims(roi_normalized, -1), 0)

            try:
                predictions = emotion_model.predict(roi_input, verbose=0)
                emotion_label = emotion_dict[int(np.argmax(predictions))]

                timestamp = datetime.now().strftime('%H:%M:%S')
                emotion_log.append({'Time': timestamp, 'Emotion': emotion_label})

                cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 2)
                cv2.putText(frame, emotion_label, (x+5, y-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            except Exception as pred_error:
                print("⚠️ Prediction error:", pred_error)

        cv2.imshow("Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 Quitting...")
            break
finally:
    cap.release()
    cv2.destroyAllWindows()

# Save log to CSV
output_file = "/home/nandkumar/Desktop/Research Projet/emotion_log.csv"
os.makedirs(os.path.dirname(output_file), exist_ok=True)

if emotion_log:
    df = pd.DataFrame(emotion_log)
    try:
        df.to_csv(output_file, index=False)
        print(f"📁 Emotion log saved: {output_file}")
    except Exception as e:
        print(f"❌ Error saving CSV: {e}")
else:
    print("ℹ️ No emotions were detected to save.")
