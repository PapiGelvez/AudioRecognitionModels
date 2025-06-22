# Script para analizar la longitud de los MFCCs de un conjunto de audios

import librosa
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve().parent

audio_files_path = project_root / "AUDIO" / "SmallDataset"

mfcc_lengths = []

folders = os.listdir(audio_files_path)

for folder in folders:
    files = os.listdir(os.path.join(audio_files_path, folder))
    for file in files:
        file_path = os.path.join(audio_files_path, folder, file)
        audio, sr = librosa.load(file_path, sr=22050)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_lengths.append(mfcc.shape[1])  # Num de frames

# Mostrar histograma
plt.hist(mfcc_lengths, bins=30, edgecolor="black")
plt.xlabel("Número de frames (tiempo)")
plt.ylabel("Cantidad de audios")
plt.title("Distribución de longitud de MFCCs")
plt.grid(True)
plt.show()

# Mostrar estadísticas básicas
print(f"Min: {np.min(mfcc_lengths)}")
print(f"Max: {np.max(mfcc_lengths)}")
print(f"Media: {np.mean(mfcc_lengths):.2f}")
print(f"Mediana: {np.median(mfcc_lengths)}")
print(f"Percentil 95: {np.percentile(mfcc_lengths, 95)}")
