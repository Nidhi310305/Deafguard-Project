from pathlib import Path
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

DATA_FILE = Path("train") / "yamnet_embeddings.npz"
MODEL_FILE = Path("train") / "deafguard_sound_classifier.joblib"
LABELS_FILE = Path("train") / "class_names.json"

loaded = np.load(DATA_FILE, allow_pickle=True)
X = loaded["embeddings"]
y = loaded["labels"].astype(str)

print(f"Features: {X.shape}")
print(f"Samples: {len(y)}")
print("Classes:")
for label in sorted(set(y)):
    print(f"  {label}: {(y == label).sum()}")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )),
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\nEvaluation results")
print(f"Test accuracy: {accuracy_score(y_test, y_pred):.3f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))

classes = sorted(set(y))
print("Confusion matrix label order:")
print(classes)
print(confusion_matrix(y_test, y_pred, labels=classes))

joblib.dump(model, MODEL_FILE)
LABELS_FILE.write_text(json.dumps(classes, indent=2), encoding="utf-8")

print(f"\nSaved model: {MODEL_FILE}")
print(f"Saved class names: {LABELS_FILE}")
