## YogaPoseClassifier_with_Accuarcy_Prediciton.
This project, YogaPoseClassifier_with_Accuracy_Prediction, uses MediaPipe to extract human body landmarks from yoga pose images, videos and webcam, and classifies the poses using a RandomForestClassifier trained on pose landmark features. Also it predicts accuracy of the poses and displays it on screen along with the pose name.
## InternshipProject
 
   	├── models/                   # Trained model files (.pkl)
   	├── data/                     # CSV landmark dataset, test videos
   	├── images/                   # Training images organized by pose label
   	├── image_testing/            # Folder with test images
   	├── video_testing/            # Folder with test videos
		├── scripts/                  # Contains programming scripts
   		├── extract_landmarks.py    # Extracts pose landmarks from images
   		├── train_model.py          # Trains and evaluates the RandomForestclassifier
		  ├── video_testing.py        # Tests on videos with pose prediction
   		├── image_testing.py        # Tests on images with pose prediction
		  ├── webcam_testing.py       # Real-time webcam-based pose classification
			├── visualize_result.py     # Creates confusion matrix and F-1 score graph
	 		├── flip.py                 # flips given training image horizontally to be included in training dataset.
		├── redundant_model           # Contains pose_classifier.pkl (not currently used since test_model.pkl is being used)
		├── results										# Contains confusion atrix and F-1 score graph
		├── Working_Demo							# Contains working demo video of the project.

## Features:
	-Uses MediaPipe Pose to extract 33 body landmarks.
	-Each pose is represented using 132 features: (x, y, z, visibility) × 33.
	-Pose classifier is trained using RandomForestClassifier.
	-Displays pose name and accuracy on screen.

## Notes:
	-Landmark normalization was tested but found to reduce accuracy, so current model uses raw landmark coordinates.
	-Accuracy ranges from 80–100% for clean, full-body yoga pose images.

