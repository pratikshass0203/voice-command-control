import librosa
import numpy as np

# ==========================================
# STEP 4: PREPARE MFCC FOR CNN
# ==========================================

# Load audio
audio, sample_rate = librosa.load(
    "my_voice.wav",
    sr=16000,
    mono=True
)

print("Audio loaded!")
print("Audio shape:", audio.shape)

# Extract MFCC
mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sample_rate,
    n_mfcc=13
)

print("MFCC shape:", mfcc.shape)

# Add CNN dimensions
# Current shape: (13, 63)
# Add channel dimension
mfcc = np.expand_dims(mfcc, axis=-1)

# Add batch dimension
mfcc = np.expand_dims(mfcc, axis=0)

print("\nMFCC prepared for CNN!")
print("CNN input shape:", mfcc.shape)
print("Data type:", mfcc.dtype)

print("\n===================================")
print("STEP 4 COMPLETED!")
print("===================================")
