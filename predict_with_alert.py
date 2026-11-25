from pathlib import Path
import argparse

import joblib
import numpy as np
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as hub
from scipy.signal import resample_poly

from alert_engine import AlertDecisionEngine

MODEL_FILE = Path("train") / "deafguard_sound_classifier.joblib"
TARGET_SAMPLE_RATE = 16000


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


def predict_one(waveform, yamnet, classifier):
    tensor = tf.convert_to_tensor(waveform, dtype=tf.float32)
    _, frame_embeddings, _ = yamnet(tensor)
    clip_embedding = tf.reduce_mean(frame_embeddings, axis=0).numpy()
    probabilities = classifier.predict_proba(clip_embedding.reshape(1, -1))[0]
    index = int(np.argmax(probabilities))
    label = str(classifier.classes_[index])
    confidence = float(probabilities[index])
    return label, confidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_file")
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()

    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    print("Loading saved classifier and YAMNet...")
    classifier = joblib.load(MODEL_FILE)
    yamnet = hub.load("https://tfhub.dev/google/yamnet/1" )
    waveform = load_audio_16k_mono(audio_path)
    engine = AlertDecisionEngine()

    print(f"\nProcessing {args.repeat} simulated windows from: {audio_path.name}\n")

    for window_number in range(args.repeat):
        label, confidence = predict_one(waveform, yamnet, classifier)
        decision = engine.process(
            label,
            confidence,
            now=float(window_number),
        )
        print(
            f"Window {window_number + 1}: "
            f"{label} | {confidence:.2%} | "
            f"{decision['status']} | {decision['reason']}"
        )


if __name__ == "__main__":
    main()
