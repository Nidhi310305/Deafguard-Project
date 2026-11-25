"""Visible DeafGuard sound monitor.

This is a presentation wrapper around the validated sound pipeline. It keeps
classification and AlertDecisionEngine logic equivalent to the real-time
prototype, but displays live status in a Tkinter window so the user can see
that the microphone is actively running.

Run directly:
    python deafguard_sound_monitor.py
"""

from pathlib import Path
import queue
import threading
import time
import json
from urllib.request import Request, urlopen
import tkinter as tk
from tkinter import messagebox

from alert_engine import AlertDecisionEngine
from deafguard_response import create_alert_response
from deafguard_vibration import trigger_vibration

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_FILE = PROJECT_DIR / "train" / "deafguard_sound_classifier.joblib"
DEVICE_INDEX = 1
RECORD_SAMPLE_RATE = 44100
TARGET_SAMPLE_RATE = 16000
CHANNELS = 1
WINDOW_SECONDS = 1.0
MIN_RMS = 0.00100

PHONE_SERVER_URL = "http://127.0.0.1:8765/alert"

LIVE_SYMBOLS = {
    "emergency_siren": "🚨",
    "vehicle_horn": "📯",
    "glass_breaking": "💥",
    "baby_crying": "👶",
    "door_knocking": "🚪",
    "footsteps": "👣",
    "other_unknown": "❔",
}
def send_alert_to_phone(response):
    alert_data = {
        "event": response.event,
        "severity": response.severity,
        "message": response.message,
        "symbol": LIVE_SYMBOLS.get(response.event, "⚠️"),
        "vibration": response.vibration_pattern,
        "priority": response.priority,
    }

    body = json.dumps(alert_data, ensure_ascii=False).encode("utf-8")

    request = Request(
        PHONE_SERVER_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=2) as connection:
            print("Android alert sent:", connection.read().decode("utf-8"))
    except Exception as error:
        print("Android alert delivery failed:", error)

class SoundMonitor:
    def __init__(self, root):
        self.root = root
        self.root.title("DeafGuard | Environmental Sound Awareness")
        self.root.geometry("760x560")
        self.root.minsize(700, 500)
        self.root.configure(bg="#0f172a")

        self.events = queue.Queue()
        self.stop_event = threading.Event()
        self.worker = None
        self.engine = AlertDecisionEngine()
        self.window_number = 0
        self.started_at = None

        self.status_var = tk.StringVar(value="STARTING: loading classifier and YAMNet...")
        self.rms_var = tk.StringVar(value="RMS energy: waiting")
        self.prediction_var = tk.StringVar(value="Prediction: waiting")
        self.confidence_var = tk.StringVar(value="Confidence: waiting")
        self.decision_var = tk.StringVar(value="Decision: waiting")
        self.reason_var = tk.StringVar(value="The microphone will begin after model loading.")
        self.action_var = tk.StringVar(value="Action: waiting")
        self.window_var = tk.StringVar(value="Windows processed: 0")

        self.build_ui()
        self.root.after(150, self.process_events)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.worker = threading.Thread(target=self.audio_worker, daemon=True)
        self.worker.start()

    def build_ui(self):
        header = tk.Frame(self.root, bg="#0f172a")
        header.pack(fill="x", padx=28, pady=(24, 8))
        tk.Label(header, text="DEAFGUARD", font=("Segoe UI", 22, "bold"),
                 fg="#f59e0b", bg="#0f172a").pack(anchor="w")
        tk.Label(header, text="Environmental Sound Awareness", font=("Segoe UI", 12),
                 fg="#cbd5e1", bg="#0f172a").pack(anchor="w", pady=(2, 0))

        status = tk.Frame(self.root, bg="#1e293b")
        status.pack(fill="x", padx=28, pady=(16, 10))
        tk.Label(status, textvariable=self.status_var, font=("Segoe UI", 14, "bold"),
                 fg="#fbbf24", bg="#1e293b").pack(anchor="w", padx=18, pady=16)

        metrics = tk.Frame(self.root, bg="#0f172a")
        metrics.pack(fill="x", padx=28, pady=8)
        metrics.grid_columnconfigure(0, weight=1)
        metrics.grid_columnconfigure(1, weight=1)
        self.metric(metrics, 0, "CURRENT PREDICTION", self.prediction_var, "#38bdf8")
        self.metric(metrics, 1, "CONFIDENCE", self.confidence_var, "#a78bfa")
        self.metric(metrics, 2, "MICROPHONE ENERGY", self.rms_var, "#22c55e")
        self.metric(metrics, 3, "PROCESSED WINDOWS", self.window_var, "#94a3b8")

        decision_box = tk.Frame(self.root, bg="#1e293b")
        decision_box.pack(fill="both", expand=True, padx=28, pady=(12, 10))
        tk.Label(decision_box, text="DECISION LAYER OUTPUT", font=("Segoe UI", 11, "bold"),
                 fg="#94a3b8", bg="#1e293b").pack(anchor="w", padx=18, pady=(16, 6))
        self.decision_label = tk.Label(decision_box, textvariable=self.decision_var,
                                       font=("Segoe UI", 22, "bold"), fg="#e2e8f0",
                                       bg="#1e293b")
        self.decision_label.pack(anchor="w", padx=18, pady=(2, 8))
        self.action_label = tk.Label(decision_box, textvariable=self.action_var,
                                      font=("Segoe UI", 14, "bold"), fg="#e2e8f0",
                                      bg="#1e293b", wraplength=650, justify="left")
        self.action_label.pack(anchor="w", padx=18, pady=(0, 8))
        tk.Label(decision_box, textvariable=self.reason_var, font=("Segoe UI", 11),
                 fg="#cbd5e1", bg="#1e293b", wraplength=650, justify="left").pack(
                     anchor="w", padx=18, pady=(0, 16))

        footer = tk.Frame(self.root, bg="#0f172a")
        footer.pack(fill="x", padx=28, pady=(4, 22))
        tk.Label(footer, text="Protected alert logic: 0.50 threshold • 3-window validation • 5-second cooldown",
                 font=("Segoe UI", 9), fg="#94a3b8", bg="#0f172a").pack(side="left")
        tk.Button(footer, text="Stop Monitor", command=self.stop,
                  font=("Segoe UI", 10, "bold"), fg="#ffffff", bg="#b91c1c",
                  activebackground="#991b1b", relief="flat", padx=16, pady=8).pack(side="right")

    def metric(self, parent, column, title, variable, color):
        row = 0 if column < 2 else 1
        col = column % 2
        box = tk.Frame(parent, bg="#1e293b")
        box.grid(row=row, column=col, sticky="ew", padx=6, pady=6)
        tk.Label(box, text=title, font=("Segoe UI", 9, "bold"), fg="#94a3b8",
                 bg="#1e293b").pack(anchor="w", padx=14, pady=(10, 2))
        tk.Label(box, textvariable=variable, font=("Segoe UI", 12, "bold"), fg=color,
                 bg="#1e293b").pack(anchor="w", padx=14, pady=(0, 10))

    @staticmethod
    def prepare_audio(audio):
        audio = np.asarray(audio)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32).reshape(-1)
        audio = resample_poly(audio, TARGET_SAMPLE_RATE, RECORD_SAMPLE_RATE).astype(np.float32)
        return np.clip(audio, -1.0, 1.0)

    @staticmethod
    def predict_audio(waveform, yamnet, classifier):
        tensor = tf.convert_to_tensor(waveform, dtype=tf.float32)
        _, frame_embeddings, _ = yamnet(tensor)
        embedding = tf.reduce_mean(frame_embeddings, axis=0).numpy()
        probabilities = classifier.predict_proba(embedding.reshape(1, -1))[0]
        index = int(np.argmax(probabilities))
        return str(classifier.classes_[index]), float(probabilities[index])

    def audio_worker(self):
        try:
            # Heavy ML/audio imports are intentionally delayed until after the
            # Tkinter window is visible. This makes startup status immediate.
            self.events.put(("status", "STARTING: preparing audio libraries..."))
            global np, sd, tf, resample_poly
            import joblib
            import numpy as np
            import sounddevice as sd
            import tensorflow as tf
            import tensorflow_hub as hub
            from scipy.signal import resample_poly

            if not MODEL_FILE.exists():
                raise FileNotFoundError(f"Missing classifier: {MODEL_FILE}")
            self.events.put(("status", "STARTING: loading classifier..."))
            classifier = joblib.load(MODEL_FILE)
            self.events.put(("status", "STARTING: loading YAMNet audio model..."))
            yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
            self.started_at = time.monotonic()
            self.events.put(("status", f"RUNNING: microphone device {DEVICE_INDEX} is active"))

            while not self.stop_event.is_set():
                audio = sd.rec(int(RECORD_SAMPLE_RATE * WINDOW_SECONDS),
                               samplerate=RECORD_SAMPLE_RATE, channels=CHANNELS,
                               dtype="float32", device=DEVICE_INDEX)
                sd.wait()
                if self.stop_event.is_set():
                    break

                self.events.put(("processing", self.window_number + 1))
                waveform = self.prepare_audio(audio)
                rms = float(np.sqrt(np.mean(np.square(waveform))))
                if rms < MIN_RMS:
                    label, confidence, gate_note = "other_unknown", 1.0, "quiet-window gate"
                else:
                    label, confidence = self.predict_audio(waveform, yamnet, classifier)
                    gate_note = "classifier"

                self.window_number += 1
                decision = self.engine.process(label, confidence,
                                               now=self.window_number * WINDOW_SECONDS)
                self.events.put(("sample", {
                    "label": label, "confidence": confidence, "rms": rms,
                    "gate_note": gate_note, "decision": decision,
                    "window": self.window_number,
                }))
        except Exception as error:
            self.events.put(("error", str(error)))

    def process_events(self):
        try:
            while True:
                event_type, payload = self.events.get_nowait()
                if event_type == "status":
                    self.status_var.set(payload)
                elif event_type == "processing":
                    self.status_var.set(f"PROCESSING: analyzing audio window {payload}...")
                elif event_type == "error":
                    self.status_var.set("ERROR: microphone monitor stopped")
                    self.reason_var.set(payload)
                    self.decision_var.set("Decision: unavailable")
                    self.decision_label.configure(fg="#ef4444")
                elif event_type == "sample":
                    decision = payload["decision"]
                    self.status_var.set("RUNNING: microphone is actively listening")
                    self.prediction_var.set(f"Prediction: {payload['label']}")
                    self.confidence_var.set(f"Confidence: {payload['confidence']:.1%}")
                    self.rms_var.set(f"RMS energy: {payload['rms']:.5f} ({payload['gate_note']})")
                    self.window_var.set(f"Windows processed: {payload['window']}")
                    status_text = decision["status"]
                    if status_text == "FINAL_ALERT":
                        status_text = f"FINAL ALERT | {decision['severity'].upper()}"
                    self.decision_var.set(f"Decision: {status_text}")
                    self.action_var.set(f"ACTION: {decision['action']}")
                    self.reason_var.set(decision["reason"])
                    if decision["status"] == "FINAL_ALERT":
                        response = create_alert_response(
                            payload["label"],
                            payload["confidence"],
                        )
                        send_alert_to_phone(response)


                        self.action_label.configure(
                            font=("Segoe UI", 16, "bold"),
                            fg="#fecaca",
                        )

                        self.reason_var.set(
                            f"{decision['reason']} | "
                            f"{response.message} | "
                            f"Visual symbol: {response.symbol} | "
                            f"Vibration: {response.vibration_pattern}"
                        )

                        # Current prototype vibration simulation
                        trigger_vibration(response.vibration_pattern)
                    else:
                        self.action_label.configure(
                            font=("Segoe UI", 14, "bold"),
                            fg="#e2e8f0",
                        )

        except queue.Empty:
            pass
        self.root.after(150, self.process_events)

    def stop(self):
        self.stop_event.set()
        self.status_var.set("STOPPING: waiting for current audio window to finish...")

    def close(self):
        self.stop_event.set()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    SoundMonitor(root)
    root.mainloop()
