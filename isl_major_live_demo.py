"""Live 15-sign ISL demo for DeafGuard.

Requires, in the same folder:
  hand_landmarker.task
  isl_major_gesture_classifier.joblib
  isl_major_label_names.json

Controls:
  Q = quit

The demo uses a short majority-vote buffer so the displayed label is less
jittery than a single-frame prediction.
"""

from collections import Counter, deque
from pathlib import Path
import json
import time
from urllib.request import Request, urlopen

import cv2
import joblib
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_FILE = Path("isl_major_gesture_classifier.joblib")
LABEL_FILE = Path("isl_major_label_names.json")

HAND_MODEL_FILE = Path("hand_landmarker.task")
CAMERA_INDEX = 0
HISTORY_SIZE = 7
MIN_DISPLAY_CONFIDENCE = 0.55
ISL_SERVER_URL = "http://127.0.0.1:8765/alert"
ISL_SEND_COOLDOWN = 3.0


DISPLAY_TEXT = {
    "doctor": "Doctor",
    "emergency": "Emergency",
    "fall": "Fall detected.",
    "food": "Food.",
    "hello": "Hello.",
    "help": "I need help.",
    "no": "No.",
    "pain": "I am in pain. Please help me.",
    "safe": "I am safe.",
    "stop": "Stop.",
    "thank_you": "Thank you.",
    "tired": "I am tired.",
    "unsafe": "I am unsafe. Please help me.",
    "water": "I need water.",
    "yes": "Yes.",
}


ISL_SYMBOLS = {
    "doctor": "⚕️",
    "emergency": "⚠️",
    "fall": "⛔",
    "food": "🍽️",
    "hello": "👋",
    "help": "🆘",
    "no": "❌",
    "pain": "🤕",
    "safe": "🛡️",
    "stop": "✋",
    "thank_you": "🙏",
    "tired": "😴",
    "unsafe": "🚨",
    "water": "💧",
    "yes": "✅",
}


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def normalize_landmarks(hand_landmarks):
    points = np.asarray(
        [[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks],
        dtype=np.float32,
    )
    points = points - points[0].copy()
    scale = float(np.max(np.linalg.norm(points, axis=1)))
    if scale > 1e-8:
        points = points / scale
    return points.reshape(1, -1)


def draw_hand(frame, hand_landmarks):
    height, width = frame.shape[:2]
    points = []
    for landmark in hand_landmarks:
        point = (
            min(max(int(landmark.x * width), 0), width - 1),
            min(max(int(landmark.y * height), 0), height - 1),
        )
        points.append(point)
        cv2.circle(frame, point, 5, (0, 255, 0), -1)
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (255, 0, 0), 2)


def send_isl_message(label, message, confidence):
    payload = {
        "event": "isl_message",
        "severity": "NONE",
        "message": message,
        "symbol": ISL_SYMBOLS.get(label, "🗣️"),
        "vibration": "none",
        "priority": 0,
        "source": "isl",
        "label": label,
        "confidence": round(float(confidence), 4),
    }

    body = json.dumps(
        payload,
        ensure_ascii=False,
    ).encode("utf-8")

    request = Request(
        ISL_SERVER_URL,
        data=body,
        headers={
            "Content-Type": "application/json; charset=utf-8"
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=2) as response:
            print(
                "ISL message sent:",
                response.read().decode("utf-8"),
            )
    except Exception as error:
        print("ISL message delivery failed:", error)


def main():
    for path in [MODEL_FILE, LABEL_FILE, HAND_MODEL_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path.resolve()}")

    classifier = joblib.load(MODEL_FILE)
    class_names = json.loads(LABEL_FILE.read_text(encoding="utf-8"))
    history = deque(maxlen=HISTORY_SIZE)
    last_sent_label = None
    last_sent_time = 0.0

    base_options = python.BaseOptions(model_asset_path=str(HAND_MODEL_FILE.resolve()))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.65,
        min_hand_presence_confidence=0.65,
        min_tracking_confidence=0.65,
    )

    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_MSMF)
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam. Close other camera apps and retry.")

    print("DeafGuard 15-sign ISL live demo")
    print("Recognized signs:", ", ".join(class_names))
    print("Press Q to quit.")

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            success, frame = camera.read()
            if not success:
                print("Could not read webcam frame.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = landmarker.detect(mp_image)

            label = "No hand detected"
            confidence = 0.0
            display_message = "Show one sign to the camera"

            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                draw_hand(frame, hand)
                features = normalize_landmarks(hand)
                probabilities = classifier.predict_proba(features)[0]
                top_index = int(np.argmax(probabilities))
                raw_label = str(classifier.classes_[top_index])
                confidence = float(probabilities[top_index])
                history.append(raw_label)

                if history:
                    stable_label, stable_count = Counter(history).most_common(1)[0]
                    stable_ratio = stable_count / len(history)
                    if confidence >= MIN_DISPLAY_CONFIDENCE and stable_ratio >= 0.57:
                        label = stable_label
                        display_message = DISPLAY_TEXT.get(
                            stable_label,
                            stable_label,
                        )

                        now = time.time()

                        if (
                            stable_label != last_sent_label
                            and now - last_sent_time >= ISL_SEND_COOLDOWN
                        ):
                            send_isl_message(
                                stable_label,
                                display_message,
                                confidence,
                            )

                            last_sent_label = stable_label
                            last_sent_time = now

                    else:
                        label = "Uncertain"
                        display_message = "Uncertain - try your other hand or hold steady"

            cv2.rectangle(frame, (15, 15), (760, 150), (25, 25, 25), -1)
            cv2.putText(frame, f"Prediction: {label}", (30, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.90, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Confidence: {confidence:.1%}", (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, display_message, (30, 125),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "Q = quit", (frame.shape[1] - 125, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("DeafGuard ISL Live Demo", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    camera.release()
    cv2.destroyAllWindows()
    print("ISL live demo stopped.")


if __name__ == "__main__":
    main()
