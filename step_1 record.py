import sounddevice as sd
import soundfile as sf

# Audio settings
SAMPLE_RATE = 16000      # 16 kHz
DURATION = 2             # Record for 2 seconds
FILENAME = "my_voice.wav"

print("Get ready...")
input("Press ENTER and then say 'YES'...")

print("Recording...")
audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="int16"
)

sd.wait()

print("Recording finished!")

# Save as WAV
sf.write(FILENAME, audio, SAMPLE_RATE, subtype="PCM_16")

print("Saved as:", FILENAME)
