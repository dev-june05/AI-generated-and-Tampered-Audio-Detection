# Internship Project Report: AI-Generated & Tampered Audio Detection

## 1. Project Overview & Problem Statement
The rapid advancement of AI voice synthesis, text-to-speech, and deepfake audio technologies has made it increasingly difficult to distinguish real human speech from machine-generated or tampered audio. This poses significant risks for fraud, misinformation, and identity theft. 

The objective of this internship project was to build an **automatic audio authenticity verification system**. Because single-model classifiers can be brittle and susceptible to evasion, this project implements a **Dual-Model Ensemble** (combining CNN and RNN architectures) to analyze both the spatial and temporal anomalies left behind by audio generation algorithms.

## 2. System Architecture
The system is built as a full-stack web application with the following architectural components:

### 2.1. Frontend (User Interface)
- **Tech Stack**: React 18, Tailwind CSS, Vite.
- **Role**: Provides a modern, dark-themed, glassmorphic UI where users can upload an audio clip, preview the audio, and visualize the detection results.
- **Features**: Displays the generated Mel-Spectrogram and three distinct prediction cards: one for the ResNet model, one for the LSTM model, and one for the combined Ensemble result, complete with confidence percentage bars.

### 2.2. Backend (API & Inference)
- **Tech Stack**: FastAPI, Uvicorn, Python, PyTorch, Librosa.
- **Role**: Handles audio validation, feature extraction (preprocessing), and deep learning model inference.
- **Data Flow**:
  1. The user uploads an audio file (WAV/MP3).
  2. The FastAPI backend resamples the audio to 16 kHz mono, trims silence, and pads/crops the sequence to a fixed 3-second window.
  3. The backend computes a **Mel-Spectrogram** (128 mel bands) from the audio array.
  4. The spectrogram is converted into two distinct tensor shapes:
     - A `(1, 224, 224)` image tensor for the ResNet model.
     - A `(94, 128)` time-series sequence tensor for the LSTM model.
  5. The backend runs inference on both models, applies a configurable weighted average to their softmax probabilities, and returns the final ensemble prediction alongside the Base64-encoded spectrogram image.

## 3. Deep Learning Methods & Models

Instead of analyzing raw audio waveforms, the project relies on **Mel-Spectrograms**. Spectrograms represent the frequency spectrum of sound over time, effectively converting audio processing into a computer vision and time-series analysis problem. 

### 3.1. ResNet-18 (Spatial Analysis)
- **Architecture**: A customized 18-layer Residual Neural Network (CNN).
- **Purpose**: Treats the spectrogram as a 2D image to detect spatial, frequency-domain artifacts commonly introduced by AI vocoders (e.g., unnatural frequency banding, phase distortions).
- **Setup**: Accepts a `1x224x224` image. Output logits are passed through a Softmax layer to predict `[Real, AI Generated]`.

### 3.2. Bidirectional LSTM (Temporal Analysis)
- **Architecture**: A 2-layer Bidirectional Long Short-Term Memory (RNN) network with 256 hidden dimensions.
- **Purpose**: Treats the spectrogram as a sequence of frequency vectors over time to detect unnatural temporal dynamics, pacing, and unnatural transitions between phonemes.
- **Setup**: Accepts a `94x128` sequence. Bi-directional processing allows the network to learn context from both past and future audio frames.

### 3.3. Ensemble Strategy
The outputs of the ResNet and LSTM models are combined using a weighted average of their Softmax probabilities (defaulting to 50/50 weighting). This fusion approach ensures that if one model is uncertain due to specific background noise, the other model can compensate, leading to highly robust detection.

## 4. Key Milestones & Achievements

During the course of the internship, the following critical milestones were achieved:

1. **Full Dataset Training (Release in the Wild)**
   - The models were trained on a massive, real-world Kaggle dataset containing over **22,000 audio samples**.
   - **Data Split**: ~17,796 training samples, 4,449 validation samples, and 3,179 test samples.

2. **State-of-the-Art Model Accuracies**
   - **ResNet-18 Model**: Achieved **99.69%** accuracy on the test set after 25 epochs.
   - **LSTM Model**: Achieved **99.34%** accuracy on the test set after 25 epochs.
   - Both models successfully converged with minimal validation loss and demonstrated near-perfect F1-scores across both `Real` and `AI Generated` classes.

3. **Backend Integration & Deployment Readiness**
   - The trained weights (`resnet_audio_model.pth` and `lstm_audio_model.pth`) were successfully integrated into the FastAPI inference pipeline.
   - The application dynamically loads these weights into memory on startup for low-latency predictions.

4. **Developer Experience & Automation**
   - Developed an automated `setup.bat` script that provisions a Python virtual environment, installs all necessary Node.js and Python dependencies (including `scikit-learn` and `librosa`), and streamlines project onboarding.
   - Restructured the project repository to cleanly separate architectural documentation into a dedicated `docs/` folder.

## 5. Critical Analysis & Limitations

While the system achieves state-of-the-art accuracy, it is important to acknowledge its current limitations:

1. **Fixed 3-Second Audio Window (Critical Limitation)**: Currently, the preprocessing pipeline pads or crops every uploaded audio file to exactly 3 seconds. If a user uploads a 1-minute recording where the first 50 seconds are real and the last 10 seconds contain an AI voice clone, the model will only analyze the first 3 seconds and completely miss the fake portion.
2. **Lack of Visual Explainability (XAI)**: Although the system displays the spectrogram to the user, the average user cannot interpret it. The system provides confidence scores but does not highlight *why* the model made its decision.
3. **Basic Ensemble Strategy**: The current ensemble relies on a hardcoded weighted average (e.g., 50% ResNet, 50% LSTM). It does not dynamically adjust to trust one model over the other based on specific audio characteristics.
4. **Vulnerability to Compression & Noise**: Audio models trained on relatively clean datasets often experience a drop in accuracy when tested on heavily compressed audio (like WhatsApp voice notes) or audio with loud background noise.
5. **Adversarial Vulnerability**: Like many deep learning systems, the models currently lack defenses against adversarial noise—imperceptible static deliberately added to a deepfake to trick the classifier into returning a "Real" prediction.

## 6. Future Enhancements
To address the limitations outlined above and further improve the system, the following enhancements are planned:
- **Sliding-Window Inference**: Expanding the 3-second fixed window to support arbitrarily long audio files by applying a sliding window across the entire file and aggregating the predictions.
- **Explainability (Grad-CAM)**: Implementing Grad-CAM to draw a heatmap directly over the spectrogram, highlighting exactly which frequencies and timestamps the ResNet model focused on when making a "Fake" prediction.
- **Data Augmentation**: Incorporating background noise, room reverberation, and MP3 compression artifacts during training to improve the models' robustness in real-world scenarios.
- **Learned Fusion Layer**: Replacing the hardcoded ensemble weights with a Meta-Learner (a small neural network) that automatically learns which model's output to trust more under specific conditions.
- **Cloud Deployment**: Containerizing the frontend and backend using Docker for seamless deployment to platforms like AWS or Render.
