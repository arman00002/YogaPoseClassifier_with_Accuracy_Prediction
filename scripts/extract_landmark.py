import os
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

output_path = os.path.join("..", "data", "pose_landmarks.csv")
data_rows = []
image_dir = os.path.join("..", "images")

for pose_label in os.listdir(image_dir):
    pose_dir = os.path.join(image_dir, pose_label)
    image_files = os.listdir(pose_dir)
    for image_file in image_files:
        image_path = os.path.join(pose_dir, image_file)
        image = cv2.imread(image_path)
        if image is None:
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            row = []
            for ldm in landmarks:
                row.extend([ldm.x, ldm.y, ldm.z, ldm.visibility])  # MediaPipe already gives normalized coordinates

            row.append(pose_label)
            data_rows.append(row)

# Build column names
columns = []
for i in range(33):
    columns += [f'x{i}', f'y{i}', f'z{i}', f'visibility{i}']
columns.append('label')

# Save to CSV
df = pd.DataFrame(data_rows, columns=columns)
df.to_csv(output_path, index=False)

