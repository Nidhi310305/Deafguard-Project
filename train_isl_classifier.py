"""Train a lightweight three-sign ISL classifier from isl_landmarks.csv.

The script:
1. Reads MediaPipe landmark rows.
2. Normalizes each hand relative to wrist landmark 0 and hand size.
3. Creates a stratified train/test split.
4. Trains an RBF SVM with probability estimates.
5. Saves the classifier and label names for live webcam inference.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DATA_FILE = Path("isl_landmarks.csv")
MODEL_FILE = Path("isl_gesture_classifier.joblib")
LABEL_FILE = Path("isl_label_names.json")
RANDOM_STATE = 42


def normalize_landmarks(row_values):
    """Make landmarks less sensitive to hand position and camera distance."""
    points = np.asarray(row_values, dtype=np.float32).reshape(21, 3)
    wrist = points[0].copy()
    points = points - wrist
    scale = float(np.max(np.linalg.norm(points, axis=1)))
    if scale > 1e-8:
        points = points / scale
    return points.reshape(-1)


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Missing {DATA_FILE.resolve()}")

    data = pd.read_csv(DATA_FILE)
    if "label" not in data.columns:
        raise ValueError("CSV must contain a label column.")

    feature_columns = [column for column in data.columns if column != "label"]
    if len(feature_columns) != 63:
        raise ValueError(f"Expected 63 landmark features, found {len(feature_columns)}")

    labels = data["label"].astype(str).to_numpy()
    raw_features = data[feature_columns].to_numpy(dtype=np.float32)
    features = np.vstack([normalize_landmarks(row) for row in raw_features])

    class_names = sorted(np.unique(labels).tolist())
    print("Samples:", len(labels))
    print("Features:", features.shape)
    print("Classes:", class_names)
    print("Counts:")
    print(data["label"].value_counts().sort_index())

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                SVC(
                    kernel="rbf",
                    C=10.0,
                    gamma="scale",
                    probability=True,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    print("\nISL classifier evaluation")
    print("-------------------------")
    print(f"Train samples: {len(y_train)}")
    print(f"Test samples: {len(y_test)}")
    print(f"Holdout accuracy: {accuracy:.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, labels=class_names, zero_division=0))
    print("Confusion matrix label order:")
    print(class_names)
    print(confusion_matrix(y_test, predictions, labels=class_names))

    joblib.dump(model, MODEL_FILE)
    LABEL_FILE.write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    print(f"\nSaved model: {MODEL_FILE.resolve()}")
    print(f"Saved class names: {LABEL_FILE.resolve()}")


if __name__ == "__main__":
    main()
