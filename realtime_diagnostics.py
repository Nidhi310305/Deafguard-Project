from pathlib import Path

import joblib
import numpy as np
import sounddevice as sd
import tensorflow as tf
import tensorflow_hub as hub
from scipy.signal import resample_poly

MODEL_FILE = Path("train") / "deafguard_sound_classifier.joblib"
DEVICE_INDEX = 1
RECORD_SAMPLE_RATE = 44100
TARGET_SAMPLE_RATE = 16000
WINDOW_SECONDS = 1.0


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


print("Loading classifier and YAMNet...")
classifier = joblib.load(MODEL_FILE)
yamnet = hub.load("https://tfhub.dev/google/yamnet/1" )

print("\nDiagnostic capture started.")
print("Stay quiet for 10 windows, then cough naturally for 5 windows.")
print("Press Ctrl+C after the output is collected.\n")

try:
    for window_number in range(1, 16):
        audio = sd.rec(
            int(RECORD_SAMPLE_RATE * WINDOW_SECONDS),
            samplerate=RECORD_SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=DEVICE_INDEX,
        )
        sd.wait()

        waveform = prepare_audio(audio)
        rms = float(np.sqrt(np.mean(np.square(waveform))))

        tensor = tf.convert_to_tensor(waveform, dtype=tf.float32)
        _, frame_embeddings, _ = yamnet(tensor)
        embedding = tf.reduce_mean(frame_embeddings, axis=0).numpy()
        probabilities = classifier.predict_proba(
            embedding.reshape(1, -1)
        )[0]

        top_indices = np.argsort(probabilities)[::-1][:3]
        top_text = ", ".join(
            f"{classifier.classes_[i]}={probabilities[i]:.2%}"
            for i in top_indices
        )

        print(
            f"Window {window_number:02d} | RMS={rms:.5f} | {top_text}"
        )

except KeyboardInterrupt:
    print("\nDiagnostic capture stopped.")
