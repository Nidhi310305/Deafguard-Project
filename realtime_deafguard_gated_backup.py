from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import sounddevice as sd
import tensorflow as tf
import tensorflow_hub as hub
from scipy.signal import resample_poly

from alert_engine import AlertDecisionEngine

MODEL_FILE = Path("train") / "deafguard_sound_classifier.joblib"
DEVICE_INDEX = 1
RECORD_SAMPLE_RATE = 44100
TARGET_SAMPLE_RATE = 16000
CHANNELS = 1
WINDOW_SECONDS = 1.0

# Based on the quiet-room diagnostic run: quiet windows were at or below 0.00073.
# This is a prototype safeguard, not a universal microphone threshold.
MIN_RMS = 0.00100


def prepare_audio(audio):
    audio = np.asarray(audio)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32).reshape(-1)
    audio = resample_poly(
        audio,
        TARGET_SAMPLE_RATE,
        RECORD_SAMPLE_RATE,
    ).astype(np.float32)
    return np.clip(audio, -1.0, 1.0)


def predict_audio(waveform, yamnet, classifier):
    tensor = tf.convert_to_tensor(waveform, dtype=tf.float32)
    _, frame_embeddings, _ = yamnet(tensor)
    embedding = tf.reduce_mean(frame_embeddings, axis=0).numpy()
    probabilities = classifier.predict_proba(
        embedding.reshape(1, -1)
    )[0]
    index = int(np.argmax(probabilities))
    label = str(classifier.classes_[index])
    confidence = float(probabilities[index])
    return label, confidence


def main():
    print("Loading DeafGuard classifier and YAMNet...")
    classifier = joblib.load(MODEL_FILE)
    yamnet = hub.load("https://tfhub.dev/google/yamnet/1" )
    engine = AlertDecisionEngine()

    print("\nDeafGuard real-time microphone prototype with energy gate")
    print(f"Microphone device: {DEVICE_INDEX}")
    print(f"Minimum RMS gate: {MIN_RMS:.5f}")
    print("Press Ctrl+C to stop.\n")

    window_number = 0

    try:
        while True:
            window_number += 1
            audio = sd.rec(
                int(RECORD_SAMPLE_RATE * WINDOW_SECONDS),
                samplerate=RECORD_SAMPLE_RATE,
                channels=CHANNELS,
                dtype="float32",
                device=DEVICE_INDEX,
            )
            sd.wait()

            waveform = prepare_audio(audio)
            rms = float(np.sqrt(np.mean(np.square(waveform))))

            if rms < MIN_RMS:
                label = "other_unknown"
                confidence = 1.0
                gate_note = "quiet-window gate"
            else:
                label, confidence = predict_audio(
                    waveform,
                    yamnet,
                    classifier,
                )
                gate_note = "classifier"

            decision = engine.process(
                label,
                confidence,
                now=window_number * WINDOW_SECONDS,
            )

            timestamp = datetime.now().strftime("%H:%M:%S")
            print(
                f"[{timestamp}] Window {window_number:03d} | "
                f"RMS={rms:.5f} | {label:<18} | "
                f"confidence={confidence:6.2%} | "
                f"{decision['status']:<20} | {gate_note}"
            )

            if decision["status"] == "FINAL_ALERT":
                print(
                    f"  ALERT: {decision['severity'].upper()} | "
                    f"{decision['action']}"
                )

    except KeyboardInterrupt:
        print("\nDeafGuard microphone prototype stopped.")
    except Exception as error:
        print(f"\nMicrophone error: {error}")
        print("Check device 1 and Windows microphone permissions.")


if __name__ == "__main__":
    main()
