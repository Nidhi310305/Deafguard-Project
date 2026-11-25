from pathlib import Path
import csv
import numpy as np
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as hub
from scipy.signal import resample_poly

METADATA_FILE = Path("train") / "metadata.csv"
OUTPUT_FILE = Path("train") / "yamnet_embeddings.npz"
OUTPUT_METADATA = Path("train") / "embedding_metadata.csv"
TARGET_SAMPLE_RATE = 16000

print("Loading YAMNet...")
yamnet = hub.load("https://tfhub.dev/google/yamnet/1" )
print("YAMNet loaded.")


def load_audio_16k_mono(path):
    waveform, sample_rate = sf.read(str(path), always_2d=False)

    # Convert stereo audio to mono by averaging channels.
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)

    # Convert integer audio to float values in approximately [-1, 1].
    if np.issubdtype(waveform.dtype, np.integer):
        info = np.iinfo(waveform.dtype)
        waveform = waveform.astype(np.float32) / max(abs(info.min), info.max)
    else:
        waveform = waveform.astype(np.float32)

    # Resample when the original file is not 16 kHz.
    if sample_rate != TARGET_SAMPLE_RATE:
        waveform = resample_poly(
            waveform,
            TARGET_SAMPLE_RATE,
            sample_rate
        ).astype(np.float32)

    # Keep the signal in the range expected by YAMNet.
    waveform = np.clip(waveform, -1.0, 1.0)
    return waveform


with METADATA_FILE.open("r", encoding="utf-8", newline="") as file:
    metadata_rows = list(csv.DictReader(file))

embeddings = []
valid_rows = []
failed_files = []

for number, row in enumerate(metadata_rows, start=1):
    audio_path = Path(row["filepath"])
    print(f"[{number}/{len(metadata_rows)}] {audio_path.name}")

    try:
        waveform = load_audio_16k_mono(audio_path)
        waveform_tensor = tf.convert_to_tensor(waveform, dtype=tf.float32)

        # YAMNet returns frame scores, 1024-dimensional embeddings,
        # and a spectrogram. We average embeddings across audio frames.
        scores, frame_embeddings, spectrogram = yamnet(waveform_tensor)
        clip_embedding = tf.reduce_mean(frame_embeddings, axis=0).numpy()

        embeddings.append(clip_embedding)
        valid_rows.append(row)

    except Exception as error:
        failed_files.append((str(audio_path), str(error)))
        print(f"FAILED: {error}")

if not embeddings:
    raise RuntimeError("No embeddings were created.")

X = np.stack(embeddings).astype(np.float32)
labels = np.array([row["event_label"] for row in valid_rows])

np.savez_compressed(
    OUTPUT_FILE,
    embeddings=X,
    labels=labels
)

with OUTPUT_METADATA.open("w", encoding="utf-8", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=valid_rows[0].keys())
    writer.writeheader()
    writer.writerows(valid_rows)

print("\nEmbedding extraction complete.")
print(f"Successful files: {len(valid_rows)}")
print(f"Embedding matrix shape: {X.shape}")
print(f"Saved embeddings: {OUTPUT_FILE}")
print(f"Saved metadata: {OUTPUT_METADATA}")

if failed_files:
    print("\nFailed files:")
    for path, error in failed_files:
        print(path)
        print(error)
else:
    print("All files processed successfully.")
