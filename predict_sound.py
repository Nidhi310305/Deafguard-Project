from pathlib import Path
import argparse
import json

import joblib
import numpy as np
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as hub
from scipy.signal import resample_poly

MODEL_FILE = Path("train") / "deafguard_sound_classifier.joblib"
CLASS_NAMES_FILE = Path("train") / "class_names.json"
TARGET_SAMPLE_RATE = 16000
CONFIDENCE_THRESHOLD = 0.70

ENVIRONMENT = {
    "baby_crying": "indoor",
    "door_knocking": "indoor",
    "emergency_siren": "outdoor",
    "footsteps": "indoor_outdoor",
    "glass_breaking": "indoor_outdoor",
    "other_unknown": "unknown",
    "vehicle_horn": "outdoor",
}

SEVERITY = {
    "baby_crying": "high",
    "door_knocking": "medium",
    "emergency_siren": "high",
    "footsteps": "medium",
    "glass_breaking": "critical",
    "other_unknown": "none",
    "vehicle_horn": "high",
}

ACTION = {
    "baby_crying": "Check the nearby area or respond to the baby.",
    "door_knocking": "Check the door when convenient.",
    "emergency_siren": "Check surroundings and seek safety if required.",
    "footsteps": "Check surroundings if the event is unexpected.",
    "glass_breaking": "Move away from the area and seek assistance.",
    "other_unknown": "No immediate alert; continue monitoring.",
    "vehicle_horn": "Check traffic and surroundings immediately.",
}


def load_audio_16k_mono(path):
    waveform, sample_rate = sf.read(str(path), always_2d=False)

    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)

    if np.issubdtype(waveform.dtype, np.integer):
        info = np.iinfo(waveform.dtype)
        waveform = waveform.astype(np.float32) / max(abs(info.min), info.max)
    else:
        waveform = waveform.astype(np.float32)

    if sample_rate != TARGET_SAMPLE_RATE:
        waveform = resample_poly(
            waveform,
            TARGET_SAMPLE_RATE,
            sample_rate,
        ).astype(np.float32)

    return np.clip(waveform, -1.0, 1.0)


def main():
    parser = argparse.ArgumentParser(
        description="Predict one DeafGuard environmental sound WAV file."
    )
    parser.add_argument("audio_file", help="Path to a WAV file")
    args = parser.parse_args()

    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    print("Loading saved classifier and YAMNet...")
    classifier = joblib.load(MODEL_FILE)
    yamnet = hub.load("https://tfhub.dev/google/yamnet/1" )

    with CLASS_NAMES_FILE.open("r", encoding="utf-8") as file:
        class_names = json.load(file)

    waveform = load_audio_16k_mono(audio_path)
    waveform_tensor = tf.convert_to_tensor(waveform, dtype=tf.float32)

    _, frame_embeddings, _ = yamnet(waveform_tensor)
    clip_embedding = tf.reduce_mean(frame_embeddings, axis=0).numpy()
    clip_embedding = clip_embedding.reshape(1, -1)

    probabilities = classifier.predict_proba(clip_embedding)[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_label = str(classifier.classes_[predicted_index])
    confidence = float(probabilities[predicted_index])

    print("\nDeafGuard prediction")
    print("-------------------")
    print(f"File: {audio_path.name}")
    print(f"Predicted sound: {predicted_label}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Environment policy: {ENVIRONMENT[predicted_label]}")
    print(f"Severity policy: {SEVERITY[predicted_label]}")

    print("\nTop probabilities:")
    ranking = np.argsort(probabilities)[::-1]
    for index in ranking[:3]:
        print(f"  {classifier.classes_[index]}: {probabilities[index]:.2%}")

    if confidence >= CONFIDENCE_THRESHOLD and predicted_label != "other_unknown":
        print("\nAlert status: CANDIDATE ALERT")
        print(f"Recommended action: {ACTION[predicted_label]}")
        print("Note: temporal validation is still required before a final alert.")
    else:
        print("\nAlert status: NO ALERT YET")
        print("Reason: confidence is below threshold or the event is unknown.")


if __name__ == "__main__":
    main()
