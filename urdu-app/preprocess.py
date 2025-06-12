import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model

TARGET_SR = 16000         
N_MFCC = 40                
SEGMENT_LENGTH = 3        
TIME_STEPS = 100           
HOP_LENGTH_RATIO = 0.5 

# Load the trained models
lstm_model = load_model('lstm_mfcc.h5')
cnn_model = load_model('cnn_spectro.h5')

def extract_mfcc(file_path):

    audio, sr = librosa.load(file_path, sr=TARGET_SR)
    n_fft = min(len(audio), 2048) 
    hop_length = int(n_fft * HOP_LENGTH_RATIO)  
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=N_MFCC, n_fft=n_fft, hop_length=hop_length
    )

    if mfcc.shape[1] < TIME_STEPS:
        padding = np.zeros((N_MFCC, TIME_STEPS - mfcc.shape[1]))
        mfcc = np.hstack((mfcc, padding))
    else:
        mfcc = mfcc[:, :TIME_STEPS]

    return mfcc.T 

def extract_spectrogram(file_path, n_fft=2048, hop_length=512, n_mels=128):
    audio, sr = librosa.load(file_path, sr=TARGET_SR)
    
    if len(audio) < n_fft:
        pad_width = n_fft - len(audio)
        audio = np.pad(audio, (0, pad_width), mode='constant')
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    if mel_db.shape[1] > 128:
        mel_db = mel_db[:, :128]
    elif mel_db.shape[1] < 128:
        mel_db = np.pad(mel_db, ((0, 0), (0, 128 - mel_db.shape[1])), mode='constant')

    return mel_db.reshape(n_mels, 128, 1)