"""DeafGuard 15-sign ISL landmark collector using MediaPipe Tasks API.

This collector creates a new major-project dataset and does not modify the
original isl_landmarks.csv dataset.

Controls:
    1 = hello          2 = help          3 = pain
    4 = emergency      5 = doctor        6 = water
    7 = food           8 = yes           9 = no
    A = stop           B = safe          C = unsafe
    D = fall           E = tired         F = thank_you
    Q = quit
"""

from pathlib import Path
import csv
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


OUTPUT_FILE = Path("isl_major_landmarks.csv")
MODEL_FILE = Path("hand_landmarker.task")
CAMERA_INDEX = 0
SAVE_INTERVAL_SECONDS = 0.25


LABEL_KEYS = {
    ord("1"): "hello",
    ord("2"): "help",
    ord("3"): "pain",
    ord("4"): "emergency",
    ord("5"): "doctor",
    ord("6"): "water",
    ord("7"): "food",
    ord("8"): "yes",
    ord("9"): "no",
    ord("a"): "stop",
    ord("b"): "safe",
    ord("c"): "unsafe",
    ord("d"): "fall",
    ord("e"): "tired",
    ord("f"): "thank_you",
}


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def landmark_row(hand_landmarks, label):
    values = []

    for landmark in hand_landmarks:
        values.extend([
            landmark.x,
            landmark.y,
            landmark.z,
        ])

    return [label] + values


def ensure_header(path):
    if path.exists() and path.stat().st_size > 0:
        return

    header = ["label"]

    for index in range(21):
        header.extend([
            f"x{index}",
            f"y{index}",
            f"z{index}",
        ])

    with path.open("w", newline="") as file:
        csv.writer(file).writerow(header)


def draw_hand(frame, hand_landmarks):
    height, width = frame.shape[:2]
    points = []

    for landmark in hand_landmarks:
        x = min(
            max(int(landmark.x * width), 0),
            width - 1,
        )
        y = min(
            max(int(landmark.y * height), 0),
            height - 1,
        )

        points.append((x, y))
        cv2.circle(
            frame,
            (x, y),
            5,
            (0, 255, 0),
            -1,
        )

    for start, end in HAND_CONNECTIONS:
        cv2.line(
            frame,
            points[start],
            points[end],
            (255, 0, 0),
            2,
        )


def build_count_text(saved_counts):
    return " | ".join(
        f"{label}:{saved_counts[label]}"
        for label in saved_counts
    )


def main():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "Missing hand_landmarker.task. Place the official "
            "MediaPipe Hand Landmarker model beside this script."
        )

    ensure_header(OUTPUT_FILE)

    base_options = python.BaseOptions(
        model_asset_path=str(MODEL_FILE.resolve())
    )

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.65,
        min_hand_presence_confidence=0.65,
        min_tracking_confidence=0.65,
    )

    camera = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_MSMF,
    )

    if not camera.isOpened():
        raise RuntimeError(
            "Could not open webcam. Close other camera apps and retry."
        )

    saved_counts = {
        label: 0
        for label in LABEL_KEYS.values()
    }

    last_saved_time = 0.0

    print("DeafGuard 15-sign ISL landmark collector")
    print("Use the webcam window and press:")
    print("1=hello  2=help  3=pain  4=emergency  5=doctor")
    print("6=water  7=food  8=yes  9=no")
    print("A=stop  B=safe  C=unsafe  D=fall  E=tired  F=thank_you")
    print("Q=quit")
    print(f"Saving data to: {OUTPUT_FILE.resolve()}")

    try:
        with (
            vision.HandLandmarker.create_from_options(options)
            as landmarker,
            OUTPUT_FILE.open("a", newline="")
            as file,
        ):
            writer = csv.writer(file)

            while True:
                success, frame = camera.read()

                if not success:
                    print("Could not read a frame from the webcam.")
                    break

                frame = cv2.flip(frame, 1)

                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame,
                )

                result = landmarker.detect(mp_image)

                detected_hand = (
                    result.hand_landmarks[0]
                    if result.hand_landmarks
                    else None
                )

                if detected_hand:
                    draw_hand(frame, detected_hand)

                cv2.rectangle(
                    frame,
                    (10, 10),
                    (950, 125),
                    (25, 25, 25),
                    -1,
                )

                cv2.putText(
                    frame,
                    "1-9: signs | A-F: signs | Q=quit",
                    (20, 38),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                count_text = build_count_text(saved_counts)

                cv2.putText(
                    frame,
                    count_text,
                    (20, 72),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.40,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    frame,
                    "Show one complete hand, then press the label key",
                    (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                cv2.imshow(
                    "DeafGuard ISL Major Dataset Collector",
                    frame,
                )

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

                if key not in LABEL_KEYS:
                    continue

                if detected_hand is None:
                    print(
                        "No hand detected. Place one complete hand "
                        "inside the camera view."
                    )
                    continue

                now = time.time()

                if now - last_saved_time < SAVE_INTERVAL_SECONDS:
                    continue

                label = LABEL_KEYS[key]

                writer.writerow(
                    landmark_row(
                        detected_hand,
                        label,
                    )
                )

                file.flush()
                saved_counts[label] += 1
                last_saved_time = now

                print(
                    f"Saved {label}: "
                    f"{saved_counts[label]}"
                )

    finally:
        camera.release()
        cv2.destroyAllWindows()

    print("Collector stopped.")
    print(f"Saved CSV: {OUTPUT_FILE.resolve()}")
    print(f"Counts: {saved_counts}")


if __name__ == "__main__":
    main()
