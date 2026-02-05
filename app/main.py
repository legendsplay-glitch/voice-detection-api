from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from app.audio_utils import save_base64_audio
from app.model import predict_voice

API_KEY = "hackathon-secret"

app = FastAPI(title="AI Generated Voice Detection API")


# -------- Request Model matching tester --------
class AudioRequest(BaseModel):
    audio_base64: str = Field(alias="audioBase64")
    audio_format: str | None = Field(default="mp3", alias="audioFormat")
    language: str | None = "en"
    message: str | None = None

    class Config:
        populate_by_name = True


# -------- Prediction Endpoint --------
@app.post("/predict")
def predict(data: AudioRequest, x_api_key: str = Header(None)):
    """
    Detect whether voice is AI-generated or human.
    """

    # --- Authentication using x-api-key header ---
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # --- Save incoming Base64 audio to temp file ---
    audio_path = save_base64_audio(data.audio_base64)

    # --- Run ML prediction ---
    prediction, confidence, explanation = predict_voice(audio_path)

    # --- Structured JSON response ---
    return {
        "prediction": prediction,
        "confidence": float(confidence),
        "explanation": explanation,
        "language": data.language,
        "audio_format": data.audio_format,
        "status": "success",
    }


# -------- Root endpoint (prevents 404 on homepage) --------
@app.get("/")
def root():
    return {"message": "AI Voice Detection API is running"}
