import os
import numpy as np
import librosa
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout, InputLayer
from keras.utils import to_categorical
import random
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import tensorflow as tf

audio_files_path = "C:/Users/santi/PG-Local/Proyecto/AUDIO/SmallDataset"

augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    Shift(min_shift=-0.5, max_shift=0.5, p=0.5)
])

# Cargar el dataset del 10% de los audios
data, labels = [], []
folders = os.listdir(audio_files_path)
real_features, fake_features = [], []

for folder in folders:
    files = os.listdir(os.path.join(audio_files_path, folder))
    for file in tqdm(files, desc=f"Processing {folder}"):
        file_path = os.path.join(audio_files_path, folder, file)
        audio, sample_rate = librosa.load(file_path, sr=22050, res_type="kaiser_fast")

        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        
        if folder == 'Real':
            real_features.append(mfcc)
        else:
            fake_features.append(mfcc)

augmented_real = []

# Aumentar el dataset de audios reales para igualar el número de audios falsos
while len(real_features) + len(augmented_real) < len(fake_features):
    idx = random.randint(0, len(real_features) - 1)
    audio_file = os.listdir(os.path.join(audio_files_path, 'Real'))[idx]
    file_path = os.path.join(audio_files_path, 'Real', audio_file)
    audio, sample_rate = librosa.load(file_path, res_type="kaiser_fast")
    augmented_audio = augment(samples=audio, sample_rate=sample_rate)
    mfccs_aug = librosa.feature.mfcc(y=augmented_audio, sr=sample_rate, n_mfcc=13)
    augmented_real.append(mfccs_aug)

data = real_features + augmented_real + fake_features
labels = (['Real'] * (len(real_features) + len(augmented_real))) + (['Fake'] * len(fake_features))

# Truncar o padear los MFCCs (200 frames)
max_len = 200
X = []
for mfcc in data:
    # Padear
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        # Truncar
        mfcc = mfcc[:, :max_len]
    X.append(mfcc)

X = np.array(X)
X = X[..., np.newaxis]  # CNN recibe input 4D (#samples, height, width, channel): (tam, 13, 200, 1)

le = LabelEncoder()
y = le.fit_transform(labels)
y = to_categorical(y)

# Separar test de train/validation (75% train, 25% test)
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# Separar train de validation (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, random_state=42)

# Mappear pooling layers
def get_pooling_layer(name):
    if name == "max":
        return MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')
    elif name == "avg":
        return AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')

# Construir el modelo CNN
def build_cnn_model(input_shape, num_classes, hp):
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))

    # Dependiendo del número de capas convolucionales
    for i in range(hp["num_of_conv_layers"]):
        model.add(Conv2D(filters=hp["number_of_kernels"], kernel_size=hp["kernel_size"], activation='relu', padding='same'))
        model.add(get_pooling_layer(hp["pooling_layer"]))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))  # Capa densa con dropout
    model.add(Dropout(hp["dropout_rate"]))

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Hiperparámetros
hyperparameter_space = {
    "kernel_size": [(3, 3), (5, 5), (7, 7)],
    "number_of_kernels": [16, 32, 64],
    "batch_size": [4, 8, 16, 32],
    "dropout_rate": [0.3, 0.4, 0.5],
    "pooling_layer": ["max", "avg"],
    "num_of_conv_layers": [2, 3, 4]
}

best_accuracy = 0
best_model = None
best_hyperparams = None
best_metrics = {}

input_shape = X_train.shape[1:]
num_labels = y.shape[1]

for i in range(25): # Intentar 25 combinaciones aleatorias de hiperparámetros
    print(f"\nTrial {i+1}/25")
    hp = {
        k: random.choice(v) 
        for k, v in hyperparameter_space.items()
    }
    print("Trying:", hp)

    temp_model = build_cnn_model(input_shape, num_labels, hp)

    history = temp_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=hp["batch_size"],
        epochs=50,
        verbose=0
    )

    val_acc = max(history.history['val_accuracy'])
    val_loss = history.history['val_loss'][np.argmax(history.history['val_accuracy'])]
    train_acc = history.history['accuracy'][np.argmax(history.history['val_accuracy'])]
    train_loss = history.history['loss'][np.argmax(history.history['val_accuracy'])]
    print(f"Validation accuracy: {val_acc:.4f}")

    if val_acc > best_accuracy:
        best_accuracy = val_acc
        best_model = temp_model
        best_hyperparams = hp
        best_metrics = {
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "train_loss": train_loss,
            "val_loss": val_loss
        }
        print("New best model found")

print("Best model trained and saved.")
print("\nBest Hyperparameters:")
for k, v in best_hyperparams.items():
    print(f"{k}: {v}")

print("\nBest Model Metrics:")
for k, v in best_metrics.items():
    print(f"{k}: {v:.4f}")
