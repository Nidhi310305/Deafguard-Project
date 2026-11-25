from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

MODEL_FILE = Path("train") / "deafguard_sound_classifier.joblib"
TEST_FILE = Path("train") / "test_embeddings.npz"

model = joblib.load(MODEL_FILE)
data = np.load(TEST_FILE, allow_pickle=True)

X_test = data["embeddings"]
y_test = data["labels"].astype(str)

y_pred = model.predict(X_test).astype(str)
probabilities = model.predict_proba(X_test)
confidence = np.max(probabilities, axis=1)

print(f"Test samples: {len(y_test)}")
print(f"Final test accuracy: {accuracy_score(y_test, y_pred):.3f}")

print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))

classes = sorted(set(y_test))
print("Confusion matrix label order:")
print(classes)
print(confusion_matrix(y_test, y_pred, labels=classes))

threshold = 0.50
eligible = (confidence >= threshold) & (y_pred != "other_unknown")
correct_eligible = eligible & (y_pred == y_test)
false_eligible = eligible & (y_pred != y_test)

target_mask = y_test != "other_unknown"
critical_mask = np.isin(
    y_test,
    ["emergency_siren", "glass_breaking", "vehicle_horn"]
)

print("\nFrozen threshold analysis")
print(f"Threshold: {threshold:.2f}")
print(f"Candidate alerts: {int(eligible.sum())}")
print(f"Correct candidate alerts: {int(correct_eligible.sum())}")
print(f"False candidate alerts: {int(false_eligible.sum())}")
print(
    "Target recall: "
    f"{correct_eligible[target_mask].sum() / target_mask.sum():.2%}"
)
print(
    "Critical-event recall: "
    f"{correct_eligible[critical_mask].sum() / critical_mask.sum():.2%}"
)
