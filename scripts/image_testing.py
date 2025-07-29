import cv2
import mediapipe as mp
import joblib
import numpy as np
import os
import glob

# Load model and label encoder
model, le = joblib.load("../models/test_model.pkl")

# Mediapipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
mp_drawing = mp.solutions.drawing_utils

# Image folder and all image paths
image_folder = r"E:\InternshipProject\image_testing"
image_paths = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))

# Check if images are found
if not image_paths:
    print("[ERROR] No images found at the specified path.")
    exit()

cv2.namedWindow("Pose Prediction", cv2.WINDOW_NORMAL)
index = 0

while 0 <= index < len(image_paths):
    image_path = image_paths[index]
    image = cv2.imread(image_path)

    if image is None:
        print(f"[WARNING] Could not read image: {image_path}")
        index += 1
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        vector = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks]).flatten().reshape(1, -1)
        prediction = model.predict(vector)[0]
        confidence = model.predict_proba(vector)[0][prediction] * 100
        label = le.inverse_transform([prediction])[0]

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Prepare label text
        text = f"{label} ({confidence:.2f}%)"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        color = (0, 255, 0)
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

        margin = 20
        if image.shape[1] < text_width + margin:
            new_width = text_width + margin
            scale = new_width / image.shape[1]
            new_height = int(image.shape[0] * scale)
            image = cv2.resize(image, (new_width, new_height))

        cv2.putText(image, text, (10, 20 + text_height), font, font_scale, color, thickness)
    else:
        cv2.putText(image, "No pose detected", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Show image
    cv2.imshow("Pose Prediction", image)

    # Wait for key input
    key = cv2.waitKey(0)

    if key == 27:  # ESC
        break
    elif key == ord('d') or key == 83:  # Right arrow or 'd'
        index += 1
    elif key == ord('a') or key == 81:  # Left arrow or 'a'
        index -= 1
        if index < 0:
            index = 0  # Prevent going before first image

cv2.destroyAllWindows()
