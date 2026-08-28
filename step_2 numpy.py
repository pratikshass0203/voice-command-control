import soundfile as sf
import numpy as np

# Read the recorded voice
audio, sample_rate = sf.read("my_voice.wav")

print("Audio loaded successfully!")
print("Sample rate:", sample_rate)
print("Number of samples:", len(audio))
print("Data type:", audio.dtype)
print("First 10 samples:")
print(audio[:10])

# Convert to NumPy array
audio = np.array(audio)

print("NumPy shape:", audio.shape)
