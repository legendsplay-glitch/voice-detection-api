from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from app.audio_utils import save_base64_audio
from app.model import predict_voice

API_KEY = "hackathon-secret"

app = FastAPI()


class AudioRequest(BaseModel):
    audio_base64: str
    message: str | None = None


@app.post("/predict")
def predict(data: AudioRequest, authorization: str = Header(None)):

    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    audio_path = save_base64_audio(data.audio_base64)

    prediction, confidence, explanation = predict_voice(audio_path)

    return {
        "prediction": prediction,
        "confidence": float(confidence),
        "explanation": explanation,
        "status": "success",
    }
