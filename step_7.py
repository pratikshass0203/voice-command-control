import os
import pickle
import numpy as np

from scipy.io import wavfile
from scipy.fftpack import dct

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


# ==========================================
# STEP 7 - TRAIN VOICE COMMAND MODEL
# ==========================================

DATASET = "dataset"

KEYWORDS = [
    "volume_up",
    "volume_down",
    "open_calculator",
    "open_notepad",
    "mute"
]

MODEL_FILE = "voice_command_model.pkl"

SAMPLE_RATE = 16000


print("======================================")
print(" STEP 7 - VOICE MODEL TRAINING")
print("======================================")
print()


# ==========================================
# MFCC EXTRACTION
# ==========================================

def extract_mfcc(filename):

    sample_rate, audio = wavfile.read(filename)

    audio = audio.astype(np.float32)

    # Convert stereo to mono if necessary
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Normalize
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

    # Number of frames
    num_frames = 1 + int(
        np.ceil(
            (len(audio) - frame_length) /
            frame_step
        )
    )

    padded_length = (
        (num_frames - 1) * frame_step
        + frame_length
    )

    audio = np.pad(
        audio,
        (0, padded_length - len(audio))
    )

    # Create frames
    indices = (
        np.arange(frame_length)[None, :]
        +
        np.arange(
            num_frames
        )[:, None] * frame_step
    )

    frames = audio[indices]

    # Hamming window
    frames *= np.hamming(frame_length)

    # FFT
    NFFT = 512

    spectrum = np.abs(
        np.fft.rfft(frames, NFFT)
    )

    power = (
        spectrum ** 2
    ) / NFFT

    # ======================================
    # MEL FILTER BANK
    # ======================================

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

                filter_bank[
                    m - 1,
                    k
                ] = (
                    k - left
                ) / (
                    center - left
                )

        if right > center:

            for k in range(center, right):

                filter_bank[
                    m - 1,
                    k
                ] = (
                    right - k
                ) / (
                    right - center
                )

    # Filter bank energy
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

    # ======================================
    # MFCC
    # ======================================

    mfcc = dct(
        log_energy,
        type=2,
        axis=1,
        norm="ortho"
    )[:, :13]

    # Fixed-size feature vector
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


# ==========================================
# LOAD DATASET
# ==========================================

X = []
y = []

print("Checking dataset...")
print()

for keyword in KEYWORDS:

    folder = os.path.join(
        DATASET,
        keyword
    )

    if not os.path.isdir(folder):

        print("ERROR!")
        print()
        print("Missing folder:")
        print(folder)
        print()
        print("Run STEP 6 again.")
        input("\nPress ENTER to exit...")
        raise SystemExit

    files = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(".wav")
    ])

    print(
        keyword,
        "->",
        len(files),
        "recordings"
    )

    if len(files) == 0:

        print()
        print("ERROR: No WAV files found in:")
        print(folder)

        input("\nPress ENTER to exit...")
        raise SystemExit

    for filename in files:

        filepath = os.path.join(
            folder,
            filename
        )

        try:

            feature = extract_mfcc(
                filepath
            )

            X.append(feature)
            y.append(keyword)

        except Exception as error:

            print(
                "Skipped:",
                filename
            )

            print(
                "Reason:",
                error
            )


# ==========================================
# CONVERT TO NUMPY
# ==========================================

X = np.array(X)
y = np.array(y)

print()
print("--------------------------------------")
print("Total recordings:", len(X))
print("Feature size:", X.shape[1])
print("--------------------------------------")
print()


# ==========================================
# CREATE MODEL
# ==========================================

print("Creating lightweight model...")

model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),

    (
        "classifier",
        MLPClassifier(
            hidden_layer_sizes=(32,),
            activation="relu",
            solver="lbfgs",
            max_iter=1000,
            random_state=42
        )
    )
])


# ==========================================
# TRAIN
# ==========================================

print()
print("Training...")
print("Please wait...")
print()

model.fit(
    X,
    y
)

print("Training completed!")
print()


# ==========================================
# TRAINING ACCURACY
# ==========================================

prediction = model.predict(X)

accuracy = accuracy_score(
    y,
    prediction
)

print("--------------------------------------")
print(
    "Training accuracy:",
    f"{accuracy * 100:.2f}%"
)
print("--------------------------------------")
print()


# ==========================================
# SAVE MODEL
# ==========================================

with open(
    MODEL_FILE,
    "wb"
) as file:

    pickle.dump(
        model,
        file
    )


print("Model saved successfully!")
print()
print("File:")
print(MODEL_FILE)
print()

print("Commands:")
for keyword in KEYWORDS:
    print("-", keyword.replace("_", " "))

print()
print("======================================")
print(" STEP 7 COMPLETED")
print("======================================")

input("\nPress ENTER to exit...")
