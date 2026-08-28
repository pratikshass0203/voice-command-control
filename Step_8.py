import os
import pickle
import ctypes
import numpy as np
import sounddevice as sd
from scipy.fftpack import dct


# ============================================================
# STEP 8 - LIVE VOICE COMMAND CONTROL
# ============================================================

# Always use the folder where THIS Python file is located.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE = os.path.join(
    BASE_DIR,
    "voice_command_model.pkl"
)

SAMPLE_RATE = 16000
RECORD_SECONDS = 1.5

CONFIDENCE_THRESHOLD = 0.70


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("======================================")
print(" STEP 8 - VOICE COMMAND CONTROL")
print("======================================")
print()

print("Loading model...")

if not os.path.exists(MODEL_FILE):

    print()
    print("ERROR: voice_command_model.pkl not found.")
    print()
    print("Expected location:")
    print(MODEL_FILE)
    print()

    input("Press ENTER to exit...")
    raise SystemExit


try:

    with open(MODEL_FILE, "rb") as file:
        model = pickle.load(file)

except Exception as error:

    print()
    print("ERROR: Could not load the model.")
    print(error)
    print()

    input("Press ENTER to exit...")
    raise SystemExit


print("Model loaded successfully!")


# ============================================================
# MFCC EXTRACTION
# SAME PIPELINE AS STEP 7
# ============================================================

def extract_mfcc(audio, sample_rate):

    audio = audio.astype(np.float32)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    maximum = np.max(np.abs(audio))

    if maximum > 0:
        audio = audio / maximum

    # Pre-emphasis
    audio = np.append(
        audio[0],
        audio[1:] - 0.97 * audio[:-1]
    )

    # Frame settings
    frame_length = int(0.025 * sample_rate)
    frame_step = int(0.010 * sample_rate)

    if len(audio) < frame_length:

        audio = np.pad(
            audio,
            (0, frame_length - len(audio))
        )

    num_frames = 1 + int(
        np.ceil(
            (len(audio) - frame_length)
            / frame_step
        )
    )

    padded_length = (
        (num_frames - 1)
        * frame_step
        + frame_length
    )

    if len(audio) < padded_length:

        audio = np.pad(
            audio,
            (0, padded_length - len(audio))
        )

    # Create frames
    indices = (
        np.arange(frame_length)[None, :]
        +
        np.arange(num_frames)[:, None]
        * frame_step
    )

    frames = audio[indices]

    # Hamming window
    frames *= np.hamming(frame_length)

    # FFT
    NFFT = 512

    spectrum = np.abs(
        np.fft.rfft(
            frames,
            NFFT
        )
    )

    power = (
        spectrum ** 2
    ) / NFFT


    # ========================================================
    # MEL FILTER BANK
    # ========================================================

    num_filters = 26

    low_freq = 300
    high_freq = 8000

    low_mel = 2595 * np.log10(
        1 + low_freq / 700
    )

    high_mel = 2595 * np.log10(
        1 + high_freq / 700
    )

    mel_points = np.linspace(
        low_mel,
        high_mel,
        num_filters + 2
    )

    hz_points = (
        700 *
        (
            10 ** (mel_points / 2595)
            - 1
        )
    )

    bins = np.floor(
        (NFFT + 1)
        * hz_points
        / sample_rate
    ).astype(int)

    filter_bank = np.zeros(
        (
            num_filters,
            NFFT // 2 + 1
        )
    )

    for m in range(1, num_filters + 1):

        left = bins[m - 1]
        center = bins[m]
        right = bins[m + 1]

        if center > left:

            for k in range(left, center):

                filter_bank[m - 1, k] = (
                    k - left
                ) / (
                    center - left
                )

        if right > center:

            for k in range(center, right):

                filter_bank[m - 1, k] = (
                    right - k
                ) / (
                    right - center
                )


    # Filter-bank energies
    energies = np.dot(
        power,
        filter_bank.T
    )

    energies = np.maximum(
        energies,
        1e-10
    )

    log_energy = np.log(
        energies
    )


    # ========================================================
    # MFCC
    # ========================================================

    mfcc = dct(
        log_energy,
        type=2,
        axis=1,
        norm="ortho"
    )[:, :13]

    mean = np.mean(
        mfcc,
        axis=0
    )

    std = np.std(
        mfcc,
        axis=0
    )

    features = np.concatenate(
        [mean, std]
    )

    return features


# ============================================================
# WINDOWS COMMAND ACTIONS
# ============================================================

def execute_command(command):

    print()
    print("Executing:", command)


    # --------------------------------------------------------
    # VOLUME UP
    # --------------------------------------------------------

    if command == "volume_up":

        ctypes.windll.user32.keybd_event(
            0xAF,
            0,
            0,
            0
        )

        ctypes.windll.user32.keybd_event(
            0xAF,
            0,
            2,
            0
        )

        print("Volume UP - one step")


    # --------------------------------------------------------
    # VOLUME DOWN
    # --------------------------------------------------------

    elif command == "volume_down":

        ctypes.windll.user32.keybd_event(
            0xAE,
            0,
            0,
            0
        )

        ctypes.windll.user32.keybd_event(
            0xAE,
            0,
            2,
            0
        )

        print("Volume DOWN - one step")


    # --------------------------------------------------------
    # CALCULATOR
    # --------------------------------------------------------

    elif command == "open_calculator":

        os.startfile("calc.exe")

        print("Calculator opened")


    # --------------------------------------------------------
    # NOTEPAD
    # --------------------------------------------------------

    elif command == "open_notepad":

        os.startfile("notepad.exe")

        print("Notepad opened")


    # --------------------------------------------------------
    # MUTE
    # --------------------------------------------------------

    elif command == "mute":

        ctypes.windll.user32.keybd_event(
            0xAD,
            0,
            0,
            0
        )

        ctypes.windll.user32.keybd_event(
            0xAD,
            0,
            2,
            0
        )

        print("Mute toggled")


    # --------------------------------------------------------
    # UNKNOWN COMMAND
    # --------------------------------------------------------

    else:

        print("No action is configured for:", command)


# ============================================================
# SHOW COMMANDS
# ============================================================

print()
print("Commands:")
print("  Volume up")
print("  Volume down")
print("  Open calculator")
print("  Open notepad")
print("  Mute")

print()
print("======================================")


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    try:

        # ----------------------------------------------------
        # ENTER PERMISSION
        # ----------------------------------------------------

        input(
            "\nPress ENTER to start listening..."
        )


        # ----------------------------------------------------
        # RECORD
        # ----------------------------------------------------

        print()
        print("LISTENING...")
        print("Speak your command NOW!")

        audio = sd.rec(
            int(
                RECORD_SECONDS
                * SAMPLE_RATE
            ),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32"
        )

        sd.wait()

        audio = audio.flatten()


        # ----------------------------------------------------
        # SILENCE CHECK
        # ----------------------------------------------------

        level = np.max(
            np.abs(audio)
        )

        if level < 0.015:

            print()
            print("No voice detected.")
            print("Command not executed.")

            continue


        # ----------------------------------------------------
        # MFCC
        # ----------------------------------------------------

        features = extract_mfcc(
            audio,
            SAMPLE_RATE
        )

        features = features.reshape(
            1,
            -1
        )


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        probabilities = model.predict_proba(
            features
        )[0]

        best_index = np.argmax(
            probabilities
        )

        command = model.classes_[
            best_index
        ]

        confidence = probabilities[
            best_index
        ]


        print()
        print("--------------------------------------")

        print(
            "Detected:",
            command
        )

        print(
            "Confidence:",
            f"{confidence * 100:.2f}%"
        )


        # ----------------------------------------------------
        # CONFIDENCE CHECK
        # ----------------------------------------------------

        if confidence < CONFIDENCE_THRESHOLD:

            print()
            print("Command rejected.")
            print("Confidence too low.")

            continue


        # ----------------------------------------------------
        # EXECUTE COMMAND ONCE
        # ----------------------------------------------------

        execute_command(
            command
        )


        # ----------------------------------------------------
        # RETURN TO WAITING
        # ----------------------------------------------------

        print()
        print("Command completed.")
        print("Microphone stopped.")
        print("Press ENTER for the next command.")


    except KeyboardInterrupt:

        print()
        print()
        print("======================================")
        print(" Voice assistant stopped.")
        print("======================================")
        break


    except Exception as error:

        print()
        print("ERROR:")
        print(error)
        print()
        print("Program is still running.")
