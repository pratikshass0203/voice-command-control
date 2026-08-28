import os
import sounddevice as sd
import soundfile as sf

# ==========================================
# STEP 6 - RECORD KEYWORD DATASET
# ==========================================

SAMPLE_RATE = 16000
DURATION = 2

# Number of recordings for each keyword
RECORDINGS_PER_WORD = 10

# New voice commands
KEYWORDS = [
    "volume up",
    "volume down",
    "open calculator",
    "open notepad",
    "mute"
]

DATASET = "dataset"

print("===================================")
print("STEP 6 - VOICE COMMAND RECORDING")
print("===================================")

# ------------------------------------------
# Create folders
# ------------------------------------------

for word in KEYWORDS:

    folder_name = word.replace(" ", "_")

    folder = os.path.join(DATASET, folder_name)

    os.makedirs(folder, exist_ok=True)

print("\nFolders created successfully!")

# ------------------------------------------
# Record each keyword
# ------------------------------------------

for word in KEYWORDS:

    folder_name = word.replace(" ", "_")

    print("\n===================================")
    print("KEYWORD:", word.upper())
    print("===================================")

    for number in range(1, RECORDINGS_PER_WORD + 1):

        input(
            f"Recording {number}/{RECORDINGS_PER_WORD} "
            f"- Press ENTER and say '{word.upper()}'..."
        )

        print("🎤 Recording...")

        audio = sd.rec(
            int(DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32"
        )

        sd.wait()

        filename = os.path.join(
            DATASET,
            folder_name,
            f"{folder_name}_{number}.wav"
        )

        sf.write(
            filename,
            audio,
            SAMPLE_RATE
        )

        print("Saved:", filename)

print("\n===================================")
print("RECORDING COMPLETED!")
print("===================================")

print("\nTotal recordings:",
      len(KEYWORDS) * RECORDINGS_PER_WORD)

print("Keywords:", KEYWORDS)
