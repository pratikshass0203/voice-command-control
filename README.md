# 🎙️ Voice Command Recognition & Windows Control System

**Low-Latency Keyword Spotting (KWS) on the Edge** — a lightweight, real-time voice-command pipeline that recognizes predefined spoken commands and controls Windows functions, built for efficient edge/embedded deployment.

> Built for Problem Statement PS-10 (Software) by **Team SILICOREX**

## 🧩 Problem It Solves

Most voice interaction today relies on heavyweight, cloud-dependent speech recognition systems. This project builds a **compact, on-device keyword spotting system** that:
- Recognizes a small, predefined vocabulary of voice commands
- Runs entirely offline with minimal compute
- Rejects unreliable predictions to avoid false command execution
- Lays the groundwork for future embedded/FPGA-based voice control

## ⚙️ How It Works

**End-to-end pipeline:**

```
Voice Input → Audio Capture → Pre-processing → MFCC Feature Extraction
→ SVM Classification → Confidence Check → Command Action
```

1. **Audio Capture** — records speech at 16 kHz mono using `sounddevice`
2. **Pre-processing** — normalization, pre-emphasis, framing, Hamming windowing
3. **MFCC Extraction** — FFT → Mel filter bank → log energy → DCT → 13 MFCC coefficients (+ mean/std)
4. **SVM Classification** — identifies the spoken command from the feature vector
5. **Confidence Check** — executes only if prediction confidence ≥ 70%; rejects otherwise
6. **Command Action** — triggers the matching Windows action (volume up/down, mute, open Calculator, open Notepad)

## ✨ Supported Commands

| Command | Action |
|---|---|
| Volume Up | Increases system volume |
| Volume Down | Decreases system volume |
| Mute | Toggles mute |
| Calculator | Opens Windows Calculator |
| Notepad | Opens Notepad |

## 🛠️ Tech Stack

| Tool | Role |
|---|---|
| `sounddevice` | Real-time audio capture |
| `soundfile` | WAV file storage |
| `NumPy` | Numerical processing |
| `Librosa` | MFCC feature extraction |
| `SciPy` | DCT computation |
| `Scikit-learn` | SVM classifier |
| `Pickle` | Model saving/loading |
| `os` / `ctypes` | Windows app launch & volume/mute control |

## 📂 Project Structure

```
voice-command-control/
├── app/                        # Application/runtime logic
├── results/                    # Training results & outputs
├── step_1 record.py            # Audio recording
├── step_2 numpy.py             # Numerical preprocessing
├── step_3mfcc.py               # MFCC feature extraction
├── step_4prepare.py            # Dataset preparation
├── step_5 classifier.py        # SVM classifier training
├── step_6dataset.py            # Dataset handling
├── step_7.py                   # Pipeline integration
├── Step_8.py                   # Final execution / command control
├── voice_command_model.pkl     # Trained SVM model
├── .gitignore
└── LICENSE
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- A working microphone

### Installation

```bash
git clone https://github.com/your-username/kws-voice-control.git
cd kws-voice-control

pip install sounddevice soundfile numpy librosa scipy scikit-learn
```

### Usage

Run the pipeline steps in order (or use the pre-trained model directly):

```bash
python "step_1 record.py"      # Record voice samples
python "step_2 numpy.py"       # Preprocess audio
python "step_3mfcc.py"         # Extract MFCC features
python "step_4prepare.py"      # Prepare dataset
python "step_5 classifier.py"  # Train SVM classifier
python "step_6dataset.py"      # Build/organize dataset
python "step_7.py"             # Integrate pipeline
python "Step_8.py"             # Run live voice command recognition
```

A pre-trained model (`voice_command_model.pkl`) is included, so you can skip straight to `Step_8.py` to try live recognition without retraining.

Speak one of the supported commands into your microphone — if the model's confidence is ≥70%, the corresponding Windows action executes automatically.

## 📊 Design Trade-offs

- **Lightweight processing** vs. robustness to heavy background noise
- **Small, fixed vocabulary** vs. scalability to many commands
- **SVM simplicity** vs. the raw performance of deep-learning models
- **Predefined commands** vs. flexibility of general-purpose speech recognition

These trade-offs were chosen deliberately to keep the system fast, low-power, and deployable on constrained/edge hardware.

## 🔮 Future Scope

- FPGA implementation of MFCC processing
- Hardware accelerator for FFT and Mel filter bank stages
- FPGA/ASIC implementation of the classifier
- Dedicated low-power keyword-spotting hardware
- Noise-robust detection and larger command vocabulary
- Real-time, always-on edge deployment

## 📚 References

- Davis, S. & Mermelstein, P. — *Comparison of Parametric Representations for Monosyllabic Word Recognition*, IEEE Trans. ASSP, 1980
- Cortes, C. & Vapnik, V. — *Support-Vector Networks*, Machine Learning, 1995
- Chen, G., Parada, C. & Heigold, G. — *Small-Footprint Keyword Spotting Using Deep Neural Networks*, Google, IEEE ICASSP, 2014
- Rabiner, L. & Schafer, R. — *Theory and Applications of Digital Speech Processing*, Pearson, 2010
- Warden, P. — *Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition*, arXiv:1804.03209, 2018

## 👥 Team SILICOREX

- Pratiksha S S
- Sasmitha S P
- Perarasan S
- Divaker O

## 📄 License

This project is licensed under the MIT License — see the LICENSE file for details.
