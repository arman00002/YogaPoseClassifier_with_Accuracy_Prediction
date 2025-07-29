import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

y_test, y_pred = joblib.load("../models/test_predictions.pkl")
le = joblib.load("../models/label_encoder.pkl")

labels=le.classes_
cm= confusion_matrix(y_test, y_pred)

plt.figure(figsize=(60, 60))
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

report = classification_report(y_test, y_pred, target_names=labels, output_dict=True)
f1_scores=[report[label]['f1-score'] for label in labels]

plt.figure(figsize=(60,60))
sns.barplot(x=labels, y=f1_scores, palette='viridis')
plt.ylim(0,1)
plt.title("F1 Scores by Pose")
plt.xlabel("Pose")
plt.ylabel("F1 Score")
plt.tight_layout()
plt.show()

