from pathlib import Path
import numpy as np
import joblib

MODEL_FILE = Path("train") / "deafguard_sound_classifier.joblib"
VALIDATION_FILE = Path("train") / "validation_embeddings.npz"

THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
UNKNOWN_LABEL = "other_unknown"
# These are the events for which missing an alert is especially undesirable.
CRITICAL_EVENTS = {"emergency_siren", "glass_breaking", "vehicle_horn"}

model = joblib.load(MODEL_FILE)
data = np.load(VALIDATION_FILE, allow_pickle=True)
X = data["embeddings"]
y_true = data["labels"].astype(str)

probabilities = model.predict_proba(X)
y_pred = model.classes_[np.argmax(probabilities, axis=1)].astype(str)
confidence = np.max(probabilities, axis=1)

print(f"Validation samples: {len(y_true)}")
print("\nThreshold results")
print("threshold | candidate_alerts | correct_alerts | false_alerts | target_recall | critical_recall")
print("----------|------------------|----------------|--------------|---------------|----------------")

for threshold in THRESHOLDS:
    eligible = (confidence >= threshold) & (y_pred != UNKNOWN_LABEL)
    correct_alerts = eligible & (y_pred == y_true)
    false_alerts = eligible & (y_pred != y_true)
    target_mask = y_true != UNKNOWN_LABEL
    critical_mask = np.isin(y_true, list(CRITICAL_EVENTS))

    candidate_count = int(eligible.sum())
    correct_count = int(correct_alerts.sum())
    false_count = int(false_alerts.sum())

    target_recall = (
        correct_alerts[target_mask].sum() / target_mask.sum()
        if target_mask.sum() else 0.0
    )
    critical_recall = (
        correct_alerts[critical_mask].sum() / critical_mask.sum()
        if critical_mask.sum() else 0.0
    )

    print(
        f"{threshold:9.2f} | "
        f"{candidate_count:16d} | "
        f"{correct_count:14d} | "
        f"{false_count:12d} | "
        f"{target_recall:13.2%} | "
        f"{critical_recall:16.2%}"
    )

print("\nInterpretation:")
print("- candidate_alerts: predictions eligible to enter temporal validation")
print("- correct_alerts: eligible predictions whose label matches the true label")
print("- false_alerts: eligible predictions with the wrong label")
print("- target_recall: correctly alerted non-unknown events / all non-unknown events")
print("- critical_recall: correctly alerted emergency siren, glass breaking, or vehicle horn clips / all such clips")
