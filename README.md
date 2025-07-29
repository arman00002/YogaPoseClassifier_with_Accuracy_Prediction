## Yoga Pose Classification with Accuarcy Prediciton.
## This project uses MediaPipe to extract human body landmarks from yoga pose images and videos, and classifies the poses using a RandomForestClassifier trained on pose landmark features.
InternshipProject
 |──
   ├── models/                  # Trained model files (.pkl)
   ├── data/                    # CSV landmark dataset, test videos
   ├── images/                 # Training images organized by pose label
   ├── image_testing/          # Folder with test images
   ├── video_testing/          # Folder with test videos
   ├── extract_landmarks.py    # Extracts pose landmarks from images
   ├── train_model.py          # Trains and evaluates the Random Forest classifier
   ├── image_testing.py        # Tests on images with pose prediction
   ├── video_testing.py        # Tests on videos with live pose detection
   ├── webcam_testing.py       # Real-time webcam-based pose classification

## Features:
-Uses MediaPipe Pose to extract 33 body landmarks.
-Each pose is represented using 132 features: (x, y, z, visibility) × 33.
-Pose classifier is trained using RandomForestClassifier.
-Displays pose name and confidence on screen.

## Notes:
-Landmark normalization was tested but found to reduce accuracy, so current model uses raw landmark coordinates.
-Accuracy ranges from 80–100% for clean, full-body yoga pose images.

