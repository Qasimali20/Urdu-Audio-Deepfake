# Urdu Audio Deepfake Detection 🎙️🧠

This project focuses on detecting deepfake (synthetically generated) Urdu speech using deep learning. Two models—a Convolutional Neural Network (CNN) and a Long Short-Term Memory (LSTM) network—were trained to classify real vs. spoofed audio based on spectral and temporal features.

## 📌 Overview

With the rise of generative AI models like Tacotron and VITS, detecting audio deepfakes has become increasingly important—especially for low-resource languages like Urdu. This project contributes to that effort using:

- CNN model on **Mel-spectrograms**
- LSTM model on **MFCCs**

## 🧠 Models

### 🔹 CNN (Convolutional Neural Network)
- Input: 128×128 log-scaled Mel-spectrograms
- Layers: 2 Conv layers + MaxPooling + Dense + Dropout
- Output: Binary classification (Real or Spoofed)

### 🔹 LSTM (Long Short-Term Memory)
- Input: 13-dimensional MFCC sequences (zero-padded)
- Layers: LSTM (128 units) + Dense + Dropout
- Output: Binary classification

## 📁 Dataset

We use the **CSALT Urdu Deepfake Audio Dataset**, which includes:

- Bonafide speech (from real speakers)
- Spoofed speech generated via:
  - Tacotron TTS
  - VITS TTS

> Note: Dataset includes over **3,000 minutes** of Urdu audio.

### 📦 Structure

- `Bonafide Part 1`: 708 files
- `Bonafide Part 2`: 495 files
- `Tacotron`: 495 files
- `VITS`: 495 files

## ⚙️ Preprocessing

- Converted stereo to mono
- Trimmed silence
- Normalized amplitude
- Resampled & padded for consistent input shape

## 📊 Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| CNN   | 89%      | 88%       | 90%    | 89%      |
| LSTM  | 85%      | 84%       | 86%    | 85%      |

## 🖼️ Visualizations

- Mel-spectrograms of real vs. spoofed speech
- MFCC patterns of bonafide audio
- Loss & accuracy training curves
