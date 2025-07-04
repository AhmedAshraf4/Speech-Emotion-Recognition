# 🗣️ Speech Recognition using CNNs and Separable Transformers

This project focuses on speech emotion recognition using deep learning models on the CREMA-D dataset. It includes custom data augmentation, feature extraction, and the implementation of classic CNN architectures (AlexNet, VGG-16, LeNet-5) with 1D and 2D convolutions, as well as a state-of-the-art Transformer model: SepTr (Separable Transformer for Audio Spectrograms).

---

## 📁 Dataset: CREMA-D
- Full Name: Crowd-Sourced Emotional Multimodal Actors Dataset
- Total Samples: 7,442 audio-visual clips
- Labels: Six basic emotions (anger, disgust, fear, happiness, sadness, surprise)
- Used Modality: Audio only
- Train/Test Split: 80% training / 20% testing

---

## 🔄 Data Augmentation
To increase model generalization, each audio sample was augmented in multiple ways:
  - Noise injection
  - Pitch shifting
  - Time stretching (speed up / slow down)
➡️ Dataset size expanded from 7,442 to 22,326 samples.

---

## 🎵 Feature Extraction
We used librosa to extract multiple features:

For 1D models:
- Zero Crossing Rate
- Energy
- MFCC
- Chroma
- Contrast
- Mel Spectrogram (1D)

For 2D models:
- Mel Spectrogram (2D)

All features were extracted using the extract_features() function.

---

## 🧪 Data Preparation
- Applied One-Hot Encoding to labels.
- Final Split:
  - 70% Training + Validation
    - With 5% used for validation
  - 30% Testing
 
---

## 🧠 Architectures Used
1. AlexNet (1D Conv)
  - Adapted to work with 1D time series data
  - Used ReLU, Pooling, and Fully Connected layers
  - Effective at capturing local temporal patterns
2. VGG-16
  - Deep CNN with 13 conv layers + 3 dense layers
  - High accuracy, but computationally heavy
  - Used both with 1D and 2D audio features
3. LeNet-5
  - Classic shallow CNN with low parameter count
  - Fast and efficient, suitable for small models

---

## 🔬 SepTr: Separable Transformer for Spectrograms
A modern attention-based model designed to process audio spectrograms more effectively than traditional Vision Transformers.

Key Features:
- Two separate transformer blocks:
  - Vertical Transformer → attends over time axis
  - Horizontal Transformer → attends over frequency axis
- Low memory footprint
- High parameter efficiency

Code Components:
- SepTrBlock: Custom transformer block
- SeparableTr: Stacked SepTrBlocks + final classification layer

Evaluation Results:
```
| Dataset            | Accuracy |
| ------------------ | -------- |
| CREMA-D            | 70.47%   |
| Speech Commands V2 | 98.51%   |
| ESC-50             | 91.13%   |

```

---

## ⚙️ Training Setup
- Loss: Cross-entropy
- Optimizer: Adam
- Epochs: 50
- Batch Size: 4
- Preprocessing: STFT, Mel Spectrogram, SpecAugment, etc.

---

## 📈 Model Comparisons
Performed comparative experiments between:
- 1D vs 2D CNN models (AlexNet, VGG-16, LeNet-5)
- SepTr vs CNNs
- Ablation studies on SepTr (vertical-only, horizontal-only attention)

---

## 💡 Applications
- Speech Emotion Recognition
- Audio Classification
- Sound Event Detection
- Audio Tagging
- Audio Generation (via spectrogram modeling)

