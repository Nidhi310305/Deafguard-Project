from pathlib import Path
from collections import defaultdict

SPLITS = ["train", "validation", "test"]
CLASSES = [
    "baby_crying",
    "door_knocking",
    "emergency_siren",
    "footsteps",
    "glass_breaking",
    "other_unknown",
    "vehicle_horn",
]

seen = defaultdict(list)

for split in SPLITS:
    print(f"\n{split.upper()}")
    total = 0

    for class_name in CLASSES:
        folder = Path(split) / class_name
        files = sorted(folder.glob("*.wav")) if folder.exists() else []
        total += len(files)

        for file in files:
            seen[file.name].append(split)

        print(f"{class_name}: {len(files)}")

    print(f"Total {split}: {total}")

duplicates = {
    filename: locations
    for filename, locations in seen.items()
    if len(locations) > 1
}

print("\nOVERLAP CHECK")
if duplicates:
    print("ERROR: duplicate filenames appear in multiple splits")
    for filename, locations in duplicates.items():
        print(filename, locations)
else:
    print("PASS: no filename overlap detected")

print(f"Total WAV files across all splits: {len(seen)}")
