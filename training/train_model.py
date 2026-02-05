import os
import numpy as np
import librosa
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib


DATASET_PATH = "training/dataset"

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")


def extract_embedding(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)

    inputs = processor(y, sampling_rate=16000, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = wav2vec_model(**inputs)

    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding


def load_dataset():
    X, y = [], []

    for label, folder in enumerate(["human", "ai"]):
        folder_path = os.path.join(DATASET_PATH, folder)

        for file in os.listdir(folder_path):
            if file.endswith(".mp3") or file.endswith(".wav"):
                path = os.path.join(folder_path, file)
                emb = extract_embedding(path)

                X.append(emb)
                y.append(label)

    return np.array(X), np.array(y)


def main():
    print("Loading dataset...")
    X, y = load_dataset()

    print("Training classifier...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=200)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(classification_report(y_test, preds))

    joblib.dump(model, "app/voice_model.pkl")
    print("Deep model saved to app/voice_model.pkl")


if __name__ == "__main__":
    main()
