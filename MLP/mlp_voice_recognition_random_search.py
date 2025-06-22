import os
import numpy as np
import librosa
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
import random
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

audio_files_path = "C:/Users/santi/PG-Local/Proyecto/AUDIO/SmallDataset"
model_path = "mlp_random_model_13.h5"

augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    Shift(min_shift=-0.5, max_shift=0.5, p=0.5)
])

def build_model(input_shape, num_labels, num_hidden_layers, neurons, dropout):
    model = Sequential()
    model.add(Dense(neurons, input_shape=(input_shape,), activation='relu'))
    model.add(Dropout(dropout))

    for x in range(num_hidden_layers - 1):
        model.add(Dense(neurons, activation='relu'))
        model.add(Dropout(dropout))

    model.add(Dense(num_labels, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Cargar el dataset del 10% de los audios
data, labels = [], []
folders = os.listdir(audio_files_path)

real_features = []
fake_features = []

for folder in folders:
    files = os.listdir(os.path.join(audio_files_path, folder))
    for file in tqdm(files, desc=f"Processing {folder}"):
        file_path = os.path.join(audio_files_path, folder, file)
        audio, sample_rate = librosa.load(file_path, sr=22050, res_type="kaiser_fast")

        mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        mfccs_features_scaled = np.mean(mfccs_features.T, axis=0)

        if folder == 'Real':
            real_features.append(mfccs_features_scaled)
        else:
            fake_features.append(mfccs_features_scaled)

augmented_real = []

# Aumentar el dataset de audios reales para igualar el número de audios falsos
while len(real_features) + len(augmented_real) < len(fake_features):
    i = random.randint(0, len(real_features) - 1)
    audio_file = os.listdir(os.path.join(audio_files_path, 'Real'))[i]
    file_path = os.path.join(audio_files_path, 'Real', audio_file)
    audio, sample_rate = librosa.load(file_path, res_type="kaiser_fast")
    augmented_audio = augment(samples=audio, sample_rate=sample_rate)
    mfccs_aug = librosa.feature.mfcc(y=augmented_audio, sr=sample_rate, n_mfcc=13)
    augmented_real.append(mfccs_aug)

data = real_features + augmented_real + fake_features
labels = (['Real'] * (len(real_features) + len(augmented_real))) + (['Fake'] * len(fake_features))

# Codificar el dataset
feature_df = pd.DataFrame({"features": data, "class": labels})
le = LabelEncoder()
feature_df["class"] = le.fit_transform(feature_df["class"])
X = np.array(feature_df["features"].tolist())
y = np.array(feature_df["class"].tolist())
y = to_categorical(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Hiperparámetros
hyperparameter_space = {
    "num_hidden_layers": [2, 3, 4],
    "neurons_per_layer": [64, 128, 256],
    "batch_size": [4, 8, 16, 32],
    "dropout_rate": [0.3, 0.4, 0.5]
}

best_accuracy = 0
best_model = None
input_shape = X_train.shape[1]
num_labels = y.shape[1]
best_hyperparams = None
best_metrics = {}

for i in range(25):  # Intentar 25 combinaciones aleatorias de hiperparámetros (~ el 23% del espacio de búsqueda)
    print(f"\nTrial {i+1}/25 - Searching random hyperparameters...")
    hp = {
        k: random.choice(v)
        for k, v in hyperparameter_space.items()
    }
    print("Trying:", hp)

    temp_model = build_model(
        input_shape, num_labels,
        hp["num_hidden_layers"],
        hp["neurons_per_layer"],
        hp["dropout_rate"]
    )

    history = temp_model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
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

model = best_model
model.save(model_path)
print("Best model trained and saved.")
print("\nBest Hyperparameters:")
for k, v in best_hyperparams.items():
    print(f"{k}: {v}")

print("\nBest Model Metrics:")
for k, v in best_metrics.items():
    print(f"{k}: {v:.4f}")

