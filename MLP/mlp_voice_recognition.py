import os
import numpy as np
import librosa
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import random
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Paths
audio_files_path = "C:/Users/santi/PG-Local/Proyecto/AUDIO/FullDataset"
model_path = "mlp_full_model_40_v2.h5"

augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    Shift(min_shift=-0.5, max_shift=0.5, p=0.5)
])

def save_history(history, filename="mlp_accuracy_and_error_history2.png"):
    """
    Plottea accuracy/loss del entrenamiento/validación del modelo en función de los epochs

    :param history: Historial de entrenamiento del modelo
    """

    fig, axs = plt.subplots(2)

    # Crear plot de accuracy
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_xlabel("Epoch")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy evaluation")

    # Crear plot de loss
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error evaluation")

    fig.tight_layout()
    plt.savefig(filename)
    plt.close()

def save_confusion_matrix(model, X_test, y_test, filename="mlp_confusion_matrix2.png"):
    y_pred = model.predict(X_test)
    y_test_labels = np.argmax(y_test, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_test_labels, y_pred_labels)
    labels = ['Real', 'Fake']
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(filename)
    plt.close()

# Revisar si el modelo ya existe
if os.path.exists(model_path):
    model = load_model(model_path)
    print("Loaded existing model.")
else:
    data, labels = [], []
    folders = os.listdir(audio_files_path)

    real_features = []
    fake_features = []

    for folder in folders:
        files = os.listdir(os.path.join(audio_files_path, folder))
        for file in tqdm(files, desc=f"Processing {folder}"):
            file_path = os.path.join(audio_files_path, folder, file)
            audio, sample_rate = librosa.load(file_path, sr=22050, res_type="kaiser_fast")

            # Por defecto: hop_length = 512 y n_fft = 2048. 40 mfccs en vez de 20 que vienen por defecto
            mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
            mfccs_features_scaled = np.mean(mfccs_features.T, axis=0)

            if folder == 'Real':
                real_features.append(mfccs_features_scaled)
            else:
                fake_features.append(mfccs_features_scaled)

    augmented_real = []

    # Aumentar el dataset de audios reales para igualar el número de audios falsos
    while len(real_features) + len(augmented_real) < len(fake_features):
        idx = random.randint(0, len(real_features) - 1)
        audio_file = os.listdir(os.path.join(audio_files_path, 'Real'))[idx]
        file_path = os.path.join(audio_files_path, 'Real', audio_file)
        audio, sample_rate = librosa.load(file_path, res_type="kaiser_fast")
        augmented_audio = augment(samples=audio, sample_rate=sample_rate)
        mfccs_aug = librosa.feature.mfcc(y=augmented_audio, sr=sample_rate, n_mfcc=40)
        mfccs_aug_scaled = np.mean(mfccs_aug.T, axis=0)
        augmented_real.append(mfccs_aug_scaled)

    data = real_features + augmented_real + fake_features
    labels = (['Real'] * (len(real_features) + len(augmented_real))) + (['Fake'] * len(fake_features))

    # Codificar el dataset
    feature_df = pd.DataFrame({"features": data, "class": labels})
    le = LabelEncoder()
    feature_df["class"] = le.fit_transform(feature_df["class"])
    X = np.array(feature_df["features"].tolist())
    y = np.array(feature_df["class"].tolist())
    y = to_categorical(y)

    # Dividir el dataset en train y test (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Label mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

    num_labels = len(feature_df["class"].unique())
    input_shape = X_train.shape[1]

    # Crear el modelo (# de capas ocultas, neuronas y dropout)
    model = Sequential([
        Dense(256, input_shape=(input_shape,)),
        Activation("relu"),
        Dropout(0.3),

        Dense(256),
        Activation("relu"),
        Dropout(0.3),

        Dense(num_labels),
        Activation("softmax")
    ])

    # Compilar el modelo (función de pérdida, optimizador y métricas)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

    # Entrenar el modelo
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=16, epochs=150, callbacks=[early_stop], verbose=1)

    save_history(history)
    save_confusion_matrix(model, X_test, y_test)

    # Guardar el modelo
    model.save(model_path)
    print("Model trained and saved.")

# TODO: Documentar función
def detect_fake(filename):
    sound_signal, sample_rate = librosa.load(filename, res_type="kaiser_fast")
    mfcc_features = librosa.feature.mfcc(y=sound_signal, sr=sample_rate, n_mfcc=40)
    mfccs_features_scaled = np.mean(mfcc_features.T, axis=0).reshape(1, -1)
    result_array = model.predict(mfccs_features_scaled)
    result_classes = ["FAKE", "REAL"]
    result = np.argmax(result_array[0])
    print("Result:", result_classes[result])

# Ejemplos
#test_real = "C:/Users/santi/Downloads/Real voz mía.ogg"
#test_fake = "C:/Users/santi/Downloads/Fake TTS voz mía.ogg"
#detect_fake(test_real)
#detect_fake(test_fake)
