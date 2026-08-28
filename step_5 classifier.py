import os
import librosa
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import sounddevice as sd
import soundfile as sf
import subprocess

# ==========================================
# STEP 5 - KEYWORD CLASSIFIER + ACTIONS
# ==========================================

DATASET = "dataset"
SAMPLE_RATE = 16000
DURATION = 2

print("===================================")
print("STEP 5 - KEYWORD SPOTTER")
print("===================================")


# ------------------------------------------
# 1. MFCC extraction
# ------------------------------------------

def extract_mfcc(filename):

    audio, sr = librosa.load(
        filename,
        sr=SAMPLE_RATE,
        mono=True
    )

    # Force every recording to the same length
    required_length = SAMPLE_RATE * DURATION

    if len(audio) < required_length:
        audio = np.pad(
            audio,
            (0, required_length - len(audio))
        )
    else:
        audio = audio[:required_length]

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=13
    )

    return mfcc.flatten()


# ------------------------------------------
# 2. Load training data
# ------------------------------------------

X = []
y = []

for keyword in os.listdir(DATASET):

    folder = os.path.join(DATASET, keyword)

    if not os.path.isdir(folder):
        continue

    print("Loading:", keyword)

    for filename in os.listdir(folder):

        if filename.lower().endswith(".wav"):

            path = os.path.join(folder, filename)

            try:
                features = extract_mfcc(path)

                X.append(features)
                y.append(keyword)

            except Exception as e:
                print("Error:", path, e)


X = np.array(X)
y = np.array(y)

print("\nTotal training samples:", len(X))
print("Keywords:", sorted(set(y)))


# ------------------------------------------
# 3. Convert keyword names to numbers
# ------------------------------------------

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)


# ------------------------------------------
# 4. Create SVM
# ------------------------------------------

model = SVC(
    kernel="linear",
    probability=True
)


# ------------------------------------------
# 5. Train classifier
# ------------------------------------------

model.fit(X, y_encoded)

print("\nClassifier trained successfully!")


# ------------------------------------------
# 6. Record a new command
# ------------------------------------------

print("\n===================================")
print("SPEAK A COMMAND")
print("===================================")

input("Press ENTER and say your keyword...")

print("Recording...")

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32"
)

sd.wait()

sf.write(
    "command.wav",
    audio,
    SAMPLE_RATE
)

print("Recording finished.")


# ------------------------------------------
# 7. Extract MFCC from command
# ------------------------------------------

features = extract_mfcc("command.wav")

features = features.reshape(1, -1)


# ------------------------------------------
# 8. Predict keyword
# ------------------------------------------

prediction = model.predict(features)

keyword = encoder.inverse_transform(prediction)[0]

probability = model.predict_proba(features)

confidence = np.max(probability) * 100


print("\nDetected keyword:", keyword.upper())
print("Confidence: {:.2f}%".format(confidence))


# ------------------------------------------
# 9. Perform an action
# ------------------------------------------

if confidence < 50:

    print("Confidence too low.")
    print("No action performed.")

elif keyword == "yes":

    print("YES detected!")
    print("Opening Notepad...")

    subprocess.Popen(["notepad.exe"])


elif keyword == "no":

    print("NO detected!")
    print("No action performed.")


elif keyword == "up":

    print("UP detected!")
    print("You can connect this to volume-up control.")


elif keyword == "down":

    print("DOWN detected!")
    print("You can connect this to volume-down control.")


elif keyword == "left":

    print("LEFT detected!")
    print("Left command received.")


elif keyword == "right":

    print("RIGHT detected!")
    print("Right command received.")


elif keyword == "on":

    print("ON detected!")
    print("Device ON command received.")


elif keyword == "off":

    print("OFF detected!")
    print("Device OFF command received.")


elif keyword == "stop":

    print("STOP detected!")
    print("Stopping program...")
    raise SystemExit


elif keyword == "go":

    print("GO detected!")
    print("Opening browser...")

    subprocess.Popen([
        "cmd",
        "/c",
        "start",
        "https://www.google.com"
    ])


else:

    print("Keyword detected, but no action assigned.")
