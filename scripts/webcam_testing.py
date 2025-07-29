import cv2
import mediapipe as mp
import joblib
import numpy as np

model, le = joblib.load("../models/test_model.pkl")

# Mediapipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Start webcam
video = cv2.VideoCapture(0)

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    
    image = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    label, confidence = "No pose detected", 0

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        
        vector = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks]).flatten().reshape(1, -1)

        
        prediction = model.predict(vector)[0]
        confidence = model.predict_proba(vector)[0][prediction] * 100
        label = le.inverse_transform([prediction])[0]

        
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Show prediction
    cv2.putText(image, f"{label} ({confidence:.2f}%)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.imshow("Webcam Pose Detection", image)

    if cv2.waitKey(1) & 0xFF == 27:  
        break

video.release()
cv2.destroyAllWindows()
