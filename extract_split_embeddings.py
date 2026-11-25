from pathlib import Path
import argparse
import json

import numpy as np
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as hub
from scipy.signal import resample_poly

TARGET_SAMPLE_RATE = 16000
CLASSES = [
    "baby_crying",
    "door_knocking",
    "emergency_siren",
    "footsteps",
    "glass_breaking",
    "other_unknown",
    "vehicle_horn",
]


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
    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=["validation", "test"])
    args = parser.parse_args()

    split_dir = Path(args.split)
    output_file = Path("train") / f"{args.split}_embeddings.npz"
    metadata_file = Path("train") / f"{args.split}_metadata.json"

    print("Loading YAMNet...")
    yamnet = hub.load("https://tfhub.dev/google/yamnet/1" )
    print("YAMNet loaded.")

    embeddings = []
    labels = []
    filepaths = []

    audio_files = []
    for class_name in CLASSES:
        class_dir = split_dir / class_name
        audio_files.extend(sorted(class_dir.glob("*.wav")))

    for number, audio_path in enumerate(audio_files, start=1):
        label = audio_path.parent.name
        print(f"[{number}/{len(audio_files)}] {label}/{audio_path.name}")

        waveform = load_audio_16k_mono(audio_path)
        waveform_tensor = tf.convert_to_tensor(waveform, dtype=tf.float32)
        _, frame_embeddings, _ = yamnet(waveform_tensor)
        clip_embedding = tf.reduce_mean(frame_embeddings, axis=0).numpy()

        embeddings.append(clip_embedding)
        labels.append(label)
        filepaths.append(str(audio_path.resolve()))

    X = np.stack(embeddings).astype(np.float32)
    y = np.array(labels)

    np.savez_compressed(
        output_file,
        embeddings=X,
        labels=y,
    )

    metadata = {
        "split": args.split,
        "files": filepaths,
        "labels": labels,
    }
    metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nComplete.")
    print(f"Split: {args.split}")
    print(f"Files processed: {len(labels)}")
    print(f"Embedding matrix shape: {X.shape}")
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()
