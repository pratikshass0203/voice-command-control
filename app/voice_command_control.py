"""
Voice Command Control
---------------------

Five supported commands:

    calculator -> Open Windows Calculator
    notepad    -> Open Notepad
    up         -> Volume Up
    down       -> Volume Down
    mute       -> Toggle Mute

Usage:
    Press SPACE to record a command.
    Speak clearly during the recording window.

IMPORTANT:
The audio preprocessing MUST match the preprocessing used when
voice_command_model.pkl was trained.
"""

import os
import sys
import time
import subprocess
import threading
import tkinter as tk
from tkinter import ttk

import numpy as np
import sounddevice as sd
import librosa
import joblib


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_RATE = 16000

# Your previous working pipeline used approximately 2 seconds.
# Keep this synchronized with Step_8.py.
RECORD_SECONDS = 2.0

# Conservative threshold to prevent accidental actions.
CONFIDENCE_THRESHOLD = 0.60

# Difference between best and second-best prediction.
MARGIN_THRESHOLD = 0.15

# Model is located one level above app/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "voice_command_model.pkl")


# ============================================================
# COMMAND MAPPING
# ============================================================

COMMANDS = {
    "calculator": "calculator",
    "calc": "calculator",

    "notepad": "notepad",

    "up": "up",
    "volume_up": "up",
    "volume up": "up",

    "down": "down",
    "volume_down": "down",
    "volume down": "down",

    "mute": "mute",
}


# ============================================================
# LOAD MODEL
# ============================================================

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    print("\nERROR: Could not load voice_command_model.pkl")
    print(e)


# ============================================================
# AUDIO RECORDING
# ============================================================

def record_audio():
    """
    Record exactly RECORD_SECONDS of mono audio.
    """

    print("\nRecording... SPEAK NOW!")

    audio = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )

    sd.wait()

    audio = audio.flatten()

    return audio


# ============================================================
# AUDIO PREPROCESSING
# ============================================================

def extract_features(audio):
    """
    Extract MFCC features.

    IMPORTANT:
    This section must match Step_8.py exactly for maximum accuracy.

    If your Step_8.py uses different MFCC parameters, replace this
    function with the exact preprocessing from Step_8.py.
    """

    # Remove DC offset
    audio = audio - np.mean(audio)

    # Normalize safely
    max_value = np.max(np.abs(audio))

    if max_value > 0:
        audio = audio / max_value

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=40
    )

    # Statistical features
    mean = np.mean(mfcc, axis=1)
    std = np.std(mfcc, axis=1)

    features = np.concatenate([mean, std])

    return features.reshape(1, -1)


# ============================================================
# PREDICTION
# ============================================================

def predict_command(audio):
    """
    Predict the command and return:

        command, confidence, second_confidence
    """

    if model is None:
        return None, 0.0, 0.0

    features = extract_features(audio)

    try:
        probabilities = model.predict_proba(features)[0]

        indices = np.argsort(probabilities)[::-1]

        best_index = indices[0]
        second_index = indices[1] if len(indices) > 1 else indices[0]

        confidence = float(probabilities[best_index])
        second_confidence = float(probabilities[second_index])

        # Get class name
        if hasattr(model, "classes_"):
            raw_command = str(model.classes_[best_index])
        else:
            raw_command = str(best_index)

        raw_command = raw_command.strip().lower()

        return raw_command, confidence, second_confidence

    except Exception as e:
        print("\nPrediction error:")
        print(e)

        return None, 0.0, 0.0


# ============================================================
# CONFIDENCE CHECK
# ============================================================

def is_reliable(confidence, second_confidence):
    """
    Reject uncertain predictions.
    """

    if confidence < CONFIDENCE_THRESHOLD:
        return False

    margin = confidence - second_confidence

    if margin < MARGIN_THRESHOLD:
        return False

    return True


# ============================================================
# WINDOWS ACTIONS
# ============================================================

def open_calculator():
    subprocess.Popen("calc.exe")


def open_notepad():
    subprocess.Popen("notepad.exe")


def volume_up():
    try:
        from pycaw.pycaw import AudioUtilities

        devices = AudioUtilities.GetSpeakers()
        volume = devices.EndpointVolume

        current = volume.GetMasterVolumeLevelScalar()

        new_value = min(1.0, current + 0.10)

        volume.SetMasterVolumeLevelScalar(new_value, None)

    except Exception as e:
        print("Volume up error:", e)


def volume_down():
    try:
        from pycaw.pycaw import AudioUtilities

        devices = AudioUtilities.GetSpeakers()
        volume = devices.EndpointVolume

        current = volume.GetMasterVolumeLevelScalar()

        new_value = max(0.0, current - 0.10)

        volume.SetMasterVolumeLevelScalar(new_value, None)

    except Exception as e:
        print("Volume down error:", e)


def toggle_mute():
    try:
        from pycaw.pycaw import AudioUtilities

        devices = AudioUtilities.GetSpeakers()
        volume = devices.EndpointVolume

        current_mute = volume.GetMute()

        volume.SetMute(not current_mute, None)

    except Exception as e:
        print("Mute error:", e)


# ============================================================
# EXECUTE COMMAND
# ============================================================

def execute_command(command):

    command = command.lower().strip()

    # Normalize possible model labels
    command = COMMANDS.get(command, command)

    if command == "calculator":

        open_calculator()
        return "Calculator opened"

    elif command == "notepad":

        open_notepad()
        return "Notepad opened"

    elif command == "up":

        volume_up()
        return "Volume increased"

    elif command == "down":

        volume_down()
        return "Volume decreased"

    elif command == "mute":

        toggle_mute()
        return "Mute toggled"

    return "Unknown command"


# ============================================================
# GUI
# ============================================================

class VoiceControlApp:

    def __init__(self, root):

        self.root = root

        self.root.title("Voice Command Control")

        self.root.geometry("650x500")

        self.root.resizable(False, False)

        self.is_recording = False

        # -----------------------------
        # Title
        # -----------------------------

        title = tk.Label(
            root,
            text="VOICE COMMAND CONTROL",
            font=("Segoe UI", 24, "bold")
        )

        title.pack(pady=(30, 5))

        subtitle = tk.Label(
            root,
            text="Press SPACE and speak one command",
            font=("Segoe UI", 12)
        )

        subtitle.pack(pady=(0, 20))

        # -----------------------------
        # Status
        # -----------------------------

        self.status = tk.Label(
            root,
            text="READY",
            font=("Segoe UI", 22, "bold")
        )

        self.status.pack(pady=20)

        # -----------------------------
        # Confidence
        # -----------------------------

        self.confidence_label = tk.Label(
            root,
            text="Confidence: --",
            font=("Segoe UI", 12)
        )

        self.confidence_label.pack(pady=5)

        # -----------------------------
        # Result
        # -----------------------------

        self.result = tk.Label(
            root,
            text="",
            font=("Segoe UI", 16, "bold")
        )

        self.result.pack(pady=15)

        # -----------------------------
        # Button
        # -----------------------------

        self.button = tk.Button(
            root,
            text="🎤  PRESS SPACE TO SPEAK",
            font=("Segoe UI", 15, "bold"),
            padx=25,
            pady=15,
            command=self.start_recording
        )

        self.button.pack(pady=25)

        # -----------------------------
        # Commands
        # -----------------------------

        commands_text = (
            "Calculator     |     Notepad\n"
            "Up              |     Down              |     Mute"
        )

        commands = tk.Label(
            root,
            text=commands_text,
            font=("Segoe UI", 11)
        )

        commands.pack(pady=10)

        # Keyboard
        self.root.bind("<space>", self.space_pressed)

    # ========================================================
    # SPACE KEY
    # ========================================================

    def space_pressed(self, event=None):

        if not self.is_recording:

            self.start_recording()

    # ========================================================
    # RECORD
    # ========================================================

    def start_recording(self):

        if self.is_recording:
            return

        self.is_recording = True

        self.button.config(
            text="🎤  LISTENING..."
        )

        self.status.config(
            text="LISTENING..."
        )

        self.result.config(
            text=""
        )

        self.confidence_label.config(
            text="Confidence: --"
        )

        # Run recording in background
        thread = threading.Thread(
            target=self.process_voice,
            daemon=True
        )

        thread.start()

    # ========================================================
    # PROCESS
    # ========================================================

    def process_voice(self):

        try:

            audio = record_audio()

            self.root.after(
                0,
                lambda: self.status.config(
                    text="ANALYZING..."
                )
            )

            command, confidence, second_confidence = predict_command(
                audio
            )

            self.root.after(
                0,
                lambda: self.show_prediction(
                    command,
                    confidence,
                    second_confidence
                )
            )

        except Exception as e:

            print("\nApplication error:")
            print(e)

            self.root.after(
                0,
                lambda: self.show_error(str(e))
            )

    # ========================================================
    # SHOW RESULT
    # ========================================================

    def show_prediction(
        self,
        command,
        confidence,
        second_confidence
    ):

        self.confidence_label.config(
            text=f"Confidence: {confidence:.2f}"
        )

        if command is None:

            self.status.config(
                text="ERROR"
            )

            self.result.config(
                text="Could not recognize command"
            )

        elif not is_reliable(
            confidence,
            second_confidence
        ):

            self.status.config(
                text="NOT SURE"
            )

            self.result.config(
                text="Please try again"
            )

            print(
                f"Rejected: {command} "
                f"({confidence:.2f})"
            )

        else:

            try:

                message = execute_command(command)

                self.status.config(
                    text="SUCCESS"
                )

                self.result.config(
                    text=message
                )

                print(
                    f"Executed: {command} "
                    f"({confidence:.2f})"
                )

            except Exception as e:

                self.status.config(
                    text="ACTION ERROR"
                )

                self.result.config(
                    text=str(e)
                )

        self.is_recording = False

        self.button.config(
            text="🎤  PRESS SPACE TO SPEAK"
        )

    # ========================================================
    # ERROR
    # ========================================================

    def show_error(self, message):

        self.status.config(
            text="ERROR"
        )

        self.result.config(
            text=message
        )

        self.is_recording = False

        self.button.config(
            text="🎤  PRESS SPACE TO SPEAK"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 50)
    print("VOICE COMMAND CONTROL")
    print("=" * 50)

    print(f"Model: {MODEL_PATH}")
    print(f"Sample rate: {SAMPLE_RATE}")
    print(f"Recording: {RECORD_SECONDS} seconds")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")

    if model is None:

        print("\nModel could not be loaded.")
        print("Check voice_command_model.pkl")

        input("\nPress Enter to exit...")
        return

    print("\nModel loaded successfully.")
    print("\nPress SPACE and speak:")
    print("Calculator")
    print("Notepad")
    print("Up")
    print("Down")
    print("Mute")

    root = tk.Tk()

    app = VoiceControlApp(root)

    root.mainloop()


if __name__ == "__main__":
    main()
