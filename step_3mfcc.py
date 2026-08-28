import librosa
import numpy as np

# ==========================================
# STEP 3: MFCC FEATURE EXTRACTION
# ==========================================

# 1. Load your recorded voice
audio, sample_rate = librosa.load(
    "my_voice.wav",
    sr=16000,
    mono=True
)

print("===================================")
print("STEP 3 - MFCC EXTRACTION")
print("===================================")

print("Audio loaded successfully!")
print("Sample rate:", sample_rate)
print("Number of samples:", len(audio))
print("Audio shape:", audio.shape)
print("Audio data type:", audio.dtype)

# 2. Extract 13 MFCC features
mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sample_rate,
    n_mfcc=13
)

# 3. Display MFCC information
print("\nMFCC extracted successfully!")
print("MFCC shape:", mfcc.shape)

print("\nFirst 5 MFCC frames:")
print(mfcc[:, :5])

print("\n===================================")
print("STEP 3 COMPLETED!")
print("===================================")
