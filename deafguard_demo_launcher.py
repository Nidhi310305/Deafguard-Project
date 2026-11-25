"""DeafGuard hackathon demo launcher.

This launcher keeps the validated ISL and environmental sound branches in
separate processes. It provides one visible control panel without merging the
camera and microphone loops into one fragile script.

Place this file beside:
  isl_live_demo.py
  realtime_deafguard_gated.py

Run:
  python deafguard_demo_launcher.py
"""

from pathlib import Path
import subprocess
import sys
import tkinter as tk
from tkinter import messagebox

PROJECT_DIR = Path(__file__).resolve().parent
ISL_SCRIPT = PROJECT_DIR / "isl_live_demo.py"
SOUND_SCRIPT = PROJECT_DIR / "deafguard_sound_monitor.py"


class DeafGuardLauncher:
    def __init__(self, root):
        self.root = root
        self.root.title("DeafGuard | Multimodal Assistive MVP")
        self.root.geometry("760x560")
        self.root.minsize(700, 500)
        self.root.configure(bg="#0f172a")
        self.processes = {}

        self.title_font = ("Segoe UI", 22, "bold")
        self.section_font = ("Segoe UI", 14, "bold")
        self.body_font = ("Segoe UI", 10)
        self.button_font = ("Segoe UI", 11, "bold")

        self.build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.close_launcher)

    def build_ui(self):
        header = tk.Frame(self.root, bg="#0f172a")
        header.pack(fill="x", padx=28, pady=(24, 8))

        tk.Label(
            header,
            text="DEAFGUARD",
            font=self.title_font,
            fg="#38bdf8",
            bg="#0f172a",
        ).pack(anchor="w")
        tk.Label(
            header,
            text="Multimodal Assistive Awareness MVP",
            font=("Segoe UI", 12),
            fg="#cbd5e1",
            bg="#0f172a",
        ).pack(anchor="w", pady=(2, 0))
        tk.Label(
            header,
            text="Two validated modules • one visible demonstration control panel",
            font=self.body_font,
            fg="#94a3b8",
            bg="#0f172a",
        ).pack(anchor="w", pady=(5, 0))

        status_frame = tk.Frame(self.root, bg="#1e293b")
        status_frame.pack(fill="x", padx=28, pady=(16, 12))
        tk.Label(
            status_frame,
            text="SYSTEM STATUS",
            font=("Segoe UI", 10, "bold"),
            fg="#94a3b8",
            bg="#1e293b",
        ).pack(anchor="w", padx=18, pady=(12, 2))
        self.status_var = tk.StringVar(value="Ready. Choose a module to begin.")
        tk.Label(
            status_frame,
            textvariable=self.status_var,
            font=("Segoe UI", 11),
            fg="#e2e8f0",
            bg="#1e293b",
        ).pack(anchor="w", padx=18, pady=(0, 12))

        modules = tk.Frame(self.root, bg="#0f172a")
        modules.pack(fill="both", expand=True, padx=28, pady=8)
        modules.grid_columnconfigure(0, weight=1)
        modules.grid_columnconfigure(1, weight=1)

        self.create_module_card(
            modules,
            column=0,
            title="ISL COMMUNICATION",
            subtitle="Three-sign webcam proof of concept",
            details="Recognizes Hello, Help, and Pain using MediaPipe landmarks and the trained SVM classifier.",
            button_text="Open ISL Demo",
            command=self.start_isl,
            accent="#22c55e",
        )
        self.create_module_card(
            modules,
            column=1,
            title="SOUND AWARENESS",
            subtitle="Visible real-time environmental alert monitor",
            details="Shows microphone activity, predictions, confidence, temporal validation, severity, and cooldown logic.",
            button_text="Open Sound Demo",
            command=self.start_sound,
            accent="#f59e0b",
        )

        footer = tk.Frame(self.root, bg="#0f172a")
        footer.pack(fill="x", padx=28, pady=(8, 24))
        tk.Button(
            footer,
            text="Stop Running Modules",
            command=self.stop_modules,
            font=self.button_font,
            fg="#ffffff",
            bg="#b91c1c",
            activebackground="#991b1b",
            activeforeground="#ffffff",
            relief="flat",
            padx=16,
            pady=9,
            cursor="hand2",
        ).pack(side="left")
        tk.Button(
            footer,
            text="Close Launcher",
            command=self.close_launcher,
            font=self.button_font,
            fg="#0f172a",
            bg="#cbd5e1",
            activebackground="#94a3b8",
            relief="flat",
            padx=16,
            pady=9,
            cursor="hand2",
        ).pack(side="right")

    def create_module_card(self, parent, column, title, subtitle, details,
                           button_text, command, accent):
        card = tk.Frame(parent, bg="#1e293b", highlightthickness=1,
                        highlightbackground="#334155")
        card.grid(row=0, column=column, sticky="nsew", padx=7, pady=7)
        card.grid_rowconfigure(3, weight=1)

        tk.Frame(card, bg=accent, height=6).grid(row=0, column=0, sticky="ew")
        tk.Label(card, text=title, font=self.section_font, fg=accent,
                 bg="#1e293b").grid(row=1, column=0, sticky="w", padx=18, pady=(18, 2))
        tk.Label(card, text=subtitle, font=("Segoe UI", 10, "bold"),
                 fg="#e2e8f0", bg="#1e293b").grid(row=2, column=0, sticky="w", padx=18)
        tk.Label(card, text=details, font=self.body_font, fg="#cbd5e1",
                 bg="#1e293b", wraplength=280, justify="left").grid(
                     row=3, column=0, sticky="nw", padx=18, pady=(14, 18))
        tk.Button(card, text=button_text, command=command, font=self.button_font,
                  fg="#0f172a", bg=accent, activebackground=accent,
                  relief="flat", padx=12, pady=9, cursor="hand2").grid(
                      row=4, column=0, sticky="ew", padx=18, pady=(0, 20))
        card.grid_columnconfigure(0, weight=1)

    def check_script(self, script):
        if not script.exists():
            messagebox.showerror(
                "Missing script",
                f"Could not find:\n{script}\n\nPlace the launcher beside the working module.",
            )
            return False
        return True

    def start_process(self, key, script, display_name):
        if not self.check_script(script):
            return
        existing = self.processes.get(key)
        if existing and existing.poll() is None:
            self.status_var.set(f"{display_name} is already running.")
            return
        try:
            process = subprocess.Popen(
                [sys.executable, str(script)],
                cwd=str(PROJECT_DIR),
                # Hide child consoles for a cleaner judge-facing demo. The
                # webcam and sound windows remain visible. During debugging,
                # change this to CREATE_NEW_CONSOLE to see terminal logs.
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            self.processes[key] = process
            self.status_var.set(f"{display_name} opened in a separate window.")
        except Exception as error:
            messagebox.showerror("Could not start module", str(error))
            self.status_var.set("Could not start the requested module.")

    def start_isl(self):
        self.start_process("isl", ISL_SCRIPT, "ISL Communication")

    def start_sound(self):
        self.start_process("sound", SOUND_SCRIPT, "Sound Awareness")

    def stop_modules(self):
        stopped = 0
        for key, process in list(self.processes.items()):
            if process.poll() is None:
                process.terminate()
                stopped += 1
        self.status_var.set(
            f"Stopped {stopped} running module(s)." if stopped else "No modules were running."
        )

    def close_launcher(self):
        running = [p for p in self.processes.values() if p.poll() is None]
        if running:
            answer = messagebox.askyesno(
                "Modules still running",
                "Some DeafGuard modules are still running. Stop them and close?",
            )
            if not answer:
                return
            self.stop_modules()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = DeafGuardLauncher(root)
    root.mainloop()
