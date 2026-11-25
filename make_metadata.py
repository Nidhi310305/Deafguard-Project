from pathlib import Path
import csv

DATASET_ROOT = Path(r"C:\Users\ns310\OneDrive\Desktop\Deafguard_ audio\train")

ENVIRONMENT = {
    "baby_crying": "indoor",
    "footsteps": "indoor_outdoor",
    "door_knocking": "indoor",
    "vehicle_horn": "outdoor",
    "emergency_siren": "outdoor",
    "glass_breaking": "indoor_outdoor",
    "other_unknown": "unknown",
}

SEVERITY = {
    "baby_crying": "high",
    "footsteps": "medium",
    "door_knocking": "medium",
    "vehicle_horn": "high",
    "emergency_siren": "high",
    "glass_breaking": "critical",
    "other_unknown": "none",
}

EXPECTED_CLASSES = set(ENVIRONMENT.keys())
rows = []
errors = []

if not DATASET_ROOT.exists():
    raise FileNotFoundError(f"Dataset folder not found: {DATASET_ROOT}")

for class_folder in sorted(DATASET_ROOT.iterdir()):
    if not class_folder.is_dir():
        continue

    class_name = class_folder.name.lower()

    if class_name not in EXPECTED_CLASSES:
        errors.append(f"Unexpected folder: {class_folder.name}")
        continue

    for audio_file in sorted(class_folder.glob("*.wav")):
        rows.append({
            "filename": audio_file.name,
            "filepath": str(audio_file.resolve()),
            "event_label": class_name,
            "environment": ENVIRONMENT[class_name],
            "severity": SEVERITY[class_name],
        })

metadata_path = DATASET_ROOT / "metadata.csv"

with metadata_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "filename",
            "filepath",
            "event_label",
            "environment",
            "severity"
        ]
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Metadata created: {metadata_path}")
print(f"Total audio files found: {len(rows)}")

print("\nCounts by class:")
for class_name in sorted(EXPECTED_CLASSES):
    count = sum(row["event_label"] == class_name for row in rows)
    print(f"{class_name}: {count}")

if errors:
    print("\nCHECK THESE ERRORS:")
    for error in errors:
        print(error)
else:
    print("\nFolder-name verification passed.")