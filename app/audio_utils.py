import base64
import uuid


def save_base64_audio(audio_base64: str) -> str:
    audio_bytes = base64.b64decode(audio_base64)
    filename = f"/tmp/{uuid.uuid4()}.mp3"

    with open(filename, "wb") as f:
        f.write(audio_bytes)

    return filename
