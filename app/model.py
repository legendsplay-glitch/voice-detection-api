import librosa
import numpy as np
import joblib


MODEL_PATH = "app/voice_model.pkl"
model = joblib.load(MODEL_PATH)


def extract_features(audio_path: str):
    y, sr = librosa.load(audio_path, sr=None)

    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    energy = np.mean(y ** 2)

    return np.array([mfcc, zcr, flatness, energy]).reshape(1, -1)


def predict_voice(audio_path: str):
    features = extract_features(audio_path)

    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0].max()

    if pred == 1:
        return "AI_GENERATED", float(prob), "ML model detected synthetic speech"
    else:
        return "HUMAN", float(prob), "ML model detected natural speech"
