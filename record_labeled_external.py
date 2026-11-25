from pathlib import Path
import argparse

import sounddevice as sd
import soundfile as sf

parser = argparse.ArgumentParser()
parser.add_argument("label", help="for example siren or background")
parser.add_argument("number", type=int, help="recording number")
args = parser.parse_args()

OUTPUT_DIR = Path("external_tests")
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / f"{args.label}_{args.number:02d}.wav"

DEVICE_INDEX = 1
SAMPLE_RATE = 44100
CHANNELS = 1
DURATION_SECONDS = 8

print(f"Output: {OUTPUT_FILE}")
input("Press Enter, then immediately play or produce the labelled sound...")

print("Recording...")
audio = sd.rec(
    int(SAMPLE_RATE * DURATION_SECONDS),
    samplerate=SAMPLE_RATE,
    channels=CHANNELS,
    dtype="float32",
    device=DEVICE_INDEX,
)
sd.wait()

sf.write(str(OUTPUT_FILE), audio, SAMPLE_RATE)
print(f"Saved: {OUTPUT_FILE}")
