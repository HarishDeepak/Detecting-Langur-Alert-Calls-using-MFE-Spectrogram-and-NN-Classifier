# Detecting Langur Alert Calls using MFE Spectrogram and 1D CNN

Real-time bioacoustic classification system for detecting langur monkey alert calls from audio using Mel Filterbank Energy (MFE) spectrogram features and a 1D Convolutional Neural Network deployed on a low-power edge device via TensorFlow Lite.

[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Edge Impulse](https://img.shields.io/badge/Edge_Impulse-000000?style=for-the-badge)](https://edgeimpulse.com)
[![LinkedIn](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/harishdeepak/)

---

## Problem

Manual monitoring of wildlife distress calls is impractical at scale. Langur monkeys emit distinct alert calls when threatened — automating detection of these calls enables passive, real-time wildlife surveillance without continuous human presence. The challenge is to run inference on a microcontroller with severe memory and compute constraints.

**Achieved: 97.1% accuracy** in langur alert call detection (reported on Edge Impulse platform).

---

## Approach: MFE Spectrogram + 1D CNN

### Feature Extraction — Mel Filterbank Energy (MFE)

Raw audio waveforms are converted to **MFE spectrograms** — a time-frequency representation that mimics how the human auditory system perceives sound:

1. Frame audio into short overlapping windows
2. Apply FFT to each window
3. Map frequency bins onto the Mel scale (logarithmic, perceptually motivated)
4. Compute energy in each Mel filterbank
5. Output: 2D array (time frames × Mel bins) used as input features

MFE is preferred over raw MFCC for edge deployment because it is computationally simpler and preserves more spectral detail while remaining compact enough for microcontroller memory.

### Feature Flattening for 1D CNN

The 2D MFE spectrogram is flattened into a 1D feature vector (`input_length` elements) and then reshaped inside the network to `(input_length / 32, 32)` — treating time-steps as a sequence and filterbank channels as features per step. This allows standard 1D convolutions to learn temporal patterns in the spectrogram.

---

## Model Architecture

```
Input: (input_length,)  ← flattened MFE spectrogram features
  ↓
Reshape: (input_length / 32, 32)
  ↓
Conv1D(32, kernel_size=3, padding='same', activation='relu')
  ↓
MaxPooling1D(pool_size=2, strides=2)
  ↓
Conv1D(64, kernel_size=3, padding='same', activation='relu')
  ↓
MaxPooling1D(pool_size=2, strides=2)
  ↓
Conv1D(128, kernel_size=3, padding='same', activation='relu')
  ↓
MaxPooling1D(pool_size=2, strides=2)
  ↓
Flatten()
  ↓
Dropout(0.5)
  ↓
Dense(n_classes, activation='softmax')
```

**Design rationale:**
- Three Conv1D layers with doubling filter counts (32→64→128) progressively extract local then global temporal patterns
- MaxPooling with stride 2 halves the temporal resolution at each stage, reducing computation
- Dropout(0.5) before the classification head provides strong regularisation for a small dataset
- Softmax output for multi-class alert call categorisation

---

## Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam (lr=0.005, β₁=0.9, β₂=0.999) |
| Loss | Categorical Cross-Entropy |
| Batch size | 32 |
| Max epochs | 100 |
| Framework | TensorFlow / Keras via Edge Impulse |

The training script is an exported Edge Impulse training job — the pipeline (feature extraction → training → quantisation → deployment) was developed using the Edge Impulse platform.

---

## Edge Deployment (TensorFlow Lite)

After training, the model is:
1. Quantised to INT8 using TensorFlow Lite (reduces model size and inference latency)
2. Compiled for microcontroller deployment
3. Deployed for real-time inference on a low-power embedded device

The embedded system continuously processes short audio windows, runs MFE extraction, and passes features to the TFLite model — triggering an alert when langur calls are detected above a confidence threshold.

---

## Conservation Application

Langur alert calls indicate predator presence or human encroachment. Automated detection enables:
- Passive round-the-clock monitoring without field staff
- Immediate logging with timestamp and GPS (when combined with IoT infrastructure)
- Population behaviour research via call frequency analysis over time

---

## File Structure

| File | Contents |
|---|---|
| `NeuralNetworkcode` | Exported Edge Impulse TensorFlow/Keras training script |
| `Projectreport.pdf` | Full project report with methodology, dataset details, and results |

---

## Tech Stack

- **Framework**: TensorFlow 2.x / Keras, TensorFlow Lite
- **Platform**: Edge Impulse (feature extraction, training pipeline, deployment)
- **Audio features**: MFE (Mel Filterbank Energy) spectrogram
- **Deployment target**: Low-power ARM microcontroller (TFLite INT8 model)

---

## References

- Edge Impulse Documentation — https://docs.edgeimpulse.com
- Mel Filterbank Energy for keyword spotting — Pete Warden, *TinyML* (O'Reilly, 2020)
