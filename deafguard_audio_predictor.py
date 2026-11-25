from pathlib import Path
import json
import tempfile

import joblib
import numpy as np
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as hub
from scipy.signal import resample_poly
import streamlit as st


MODEL_FILE = Path("train") / "deafguard_sound_classifier.joblib"
CLASS_NAMES_FILE = Path("train") / "class_names.json"
TARGET_SAMPLE_RATE = 16000


@st.cache_resource
def load_prediction_assets():
    classifier = joblib.load(MODEL_FILE)
    yamnet = hub.load("https://tfhub.dev/google/yamnet/1" )

    with CLASS_NAMES_FILE.open("r", encoding="utf-8") as file:
        class_names = json.load(file)

    return classifier, yamnet, class_names


def load_audio_16k_mono(path):
    waveform, sample_rate = sf.read(
        str(path),
        always_2d=False,
    )

    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)

    if np.issubdtype(waveform.dtype, np.integer):
        info = np.iinfo(waveform.dtype)
        waveform = waveform.astype(np.float32)
        waveform /= max(abs(info.min), info.max)
    else:
        waveform = waveform.astype(np.float32)

    if sample_rate != TARGET_SAMPLE_RATE:
        waveform = resample_poly(
            waveform,
            TARGET_SAMPLE_RATE,
            sample_rate,
        ).astype(np.float32)

    return np.clip(waveform, -1.0, 1.0)


def predict_uploaded_wav(uploaded_file):
    classifier, yamnet, class_names = load_prediction_assets()

    audio_bytes = uploaded_file.getvalue()

    if not audio_bytes:
        raise ValueError("The uploaded WAV file is empty.")

    with tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False,
    ) as temporary_file:
        temporary_file.write(audio_bytes)
        temporary_path = Path(temporary_file.name)

    try:
        waveform = load_audio_16k_mono(temporary_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    if len(waveform) == 0:
        raise ValueError("The uploaded WAV file contains no audio samples.")

    waveform_tensor = tf.convert_to_tensor(
        waveform,
        dtype=tf.float32,
    )

    _, frame_embeddings, _ = yamnet(waveform_tensor)

    clip_embedding = tf.reduce_mean(
        frame_embeddings,
        axis=0,
    ).numpy().reshape(1, -1)

    probabilities = classifier.predict_proba(clip_embedding)[0]

    predicted_index = int(np.argmax(probabilities))
    predicted_label = str(classifier.classes_[predicted_index])
    confidence = float(probabilities[predicted_index])

    ranking = np.argsort(probabilities)[::-1]

    top_predictions = [
        {
            "label": str(classifier.classes_[index]),
            "confidence": float(probabilities[index]),
        }
        for index in ranking[:3]
    ]

    # Confirm the classifier labels are represented in the saved label file.
    if predicted_label not in class_names:
        raise ValueError(
            f"Classifier returned unknown label: {predicted_label}"
        )

    return {
        "event": predicted_label,
        "confidence": confidence,
        "top_predictions": top_predictions,
    }
