# Script para extraer características de audio y visualizarlas durante el proceso de convertir a MFCCs

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from pathlib import Path

np.complex = complex

project_root = Path(__file__).resolve().parent

# Revisar que exista la carpeta AUDIO y la subcarpeta FullDataset, o crearla si no existe con full_dataset_script.py
audio_file_path = (
    project_root
    / "AUDIO"
    / "FullDataset"
    / "Real"
    / "Real_Colombia_cof_06136_cof_06136_01979237942.wav"
)

signal, sample_rate = librosa.load(audio_file_path, sr=22050)

# Onda
plt.figure(figsize=(10, 3))
librosa.display.waveshow(signal, sr=sample_rate, alpha=0.6)
plt.title("Waveform")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.savefig("preprocess images/1_waveform.png")
plt.close()

# Framing
frame_length = int(0.025 * sample_rate)  # 25ms
hop_length = int(0.010 * sample_rate)  # 10ms

# Seleccionar un sample con audio
start_sec = 3
start_sample = int(start_sec * sample_rate)
frame_start = start_sample + hop_length * 2
frame = signal[frame_start : frame_start + frame_length]

plt.figure(figsize=(6, 3))
plt.plot(np.linspace(0, 25, frame_length), frame)
plt.title("Selected Frame (3–4s)")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.savefig("preprocess images/2_selected_frame.png")
plt.close()

# Windowing
hamming = np.hamming(frame_length)
windowed_frame = frame * hamming

plt.figure(figsize=(6, 3))
plt.plot(np.linspace(0, 25, frame_length), windowed_frame)
plt.title("Hamming Windowed Frame")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.savefig("preprocess images/3_windowed_frame.png")
plt.close()

# FFT
fft_spectrum = np.fft.rfft(windowed_frame)
magnitude_spectrum = np.abs(fft_spectrum)
freqs = np.fft.rfftfreq(frame_length, d=1 / sample_rate)

plt.figure(figsize=(6, 3))
plt.plot(freqs, magnitude_spectrum)
plt.title("FFT Magnitude Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.tight_layout()
plt.savefig("preprocess images/4_fft_spectrum.png")
plt.close()

# Mel filterbank
n_mels = 40
mel_filterbank = librosa.filters.mel(sr=sample_rate, n_fft=frame_length, n_mels=n_mels)
power_spectrum = magnitude_spectrum**2
mel_energies = np.dot(mel_filterbank, power_spectrum)

plt.figure(figsize=(6, 3))
plt.plot(mel_energies)
plt.title("Mel Filterbank Energies")
plt.xlabel("Mel Bands")
plt.ylabel("Energy")
plt.tight_layout()
plt.savefig("preprocess images/5_mel_energies.png")
plt.close()

# Log-Mel
log_mel_energies = np.log(mel_energies + 1e-10)

plt.figure(figsize=(6, 3))
plt.plot(log_mel_energies)
plt.title("Log-Mel Energies")
plt.xlabel("Mel Bands")
plt.ylabel("Log Energy")
plt.tight_layout()
plt.savefig("preprocess images/6_log_mel_energies.png")
plt.close()
