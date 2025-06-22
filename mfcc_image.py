# Script para generar una imagen de los MFCCs de un audio específico

import librosa, librosa.display
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

project_root = Path(__file__).resolve().parent

# Revisar que exista la carpeta AUDIO y la subcarpeta FullDataset, o crearla si no existe con full_dataset_script.py
audio_file_path = (
    project_root
    / "AUDIO"
    / "FullDataset"
    / "Real"
    / "Real_Colombia_cof_06136_cof_06136_01979237942.wav"
)

audio, sample_rate = librosa.load(audio_file_path, sr=22050, res_type="kaiser_fast")

mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)

plt.figure()
librosa.display.specshow(mfccs_features, sr=sample_rate, hop_length=512)
plt.xlabel("Time")
plt.ylabel("MFCC coefficients")
plt.colorbar()
plt.title("MFCCs")

plt.tight_layout()
plt.savefig("MFCC.png")
plt.close()
