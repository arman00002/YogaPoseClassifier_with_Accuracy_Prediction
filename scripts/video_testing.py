import cv2
import mediapipe as mp
import joblib
import numpy as np
import os
import glob

# Set folder for video testing
video_folder = r"E:\InternshipProject\video_testing"
video_paths = glob.glob(os.path.join(video_folder, "*.mp4"))

# Load model and label encoder
model, le = joblib.load("../models/test_model.pkl")

# Mediapipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Create a named window just once and make it resizable
cv2.namedWindow("Video Pose Detection", cv2.WINDOW_NORMAL)

# Process each video
for video_path in video_paths:
    print(f"[INFO] Processing: {os.path.basename(video_path)}")
    video = cv2.VideoCapture(video_path)

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        label, confidence = "No pose", 0

        if results.pose_landmarks and len(results.pose_landmarks.landmark) == 33:
            landmarks = results.pose_landmarks.landmark
            vector = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks]).flatten().reshape(1, -1)

            prediction = model.predict(vector)[0]
            pred_proba = model.predict_proba(vector)[0]
            confidence = pred_proba[prediction] * 100
            label = le.inverse_transform([prediction])[0]

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display pose label and confidence
        cv2.putText(frame, f"{label} ({confidence:.2f}%)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Show frame
        cv2.imshow("Video Pose Detection", frame)

        # ESC key to break out
        if cv2.waitKey(1) & 0xFF == 27:
            video.release()
            break

# Clean up
cv2.destroyAllWindows()
