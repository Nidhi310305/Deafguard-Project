from pathlib import Path
import sounddevice as sd
import soundfile as sf

OUTPUT_DIR = Path("external_tests")
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "phone_siren_recorded_by_laptop.wav"
DEVICE_INDEX = 1
SAMPLE_RATE = 44100
CHANNELS = 1
DURATION_SECONDS = 8

print("Prepare the phone siren, but do not play it yet.")
input("Press Enter, then immediately play the siren for 8 seconds...")

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
print(f"Saved recording: {OUTPUT_FILE}")

