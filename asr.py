from openai import OpenAI

from utils import get_openai_api_key


def transcribe_audio(audio_path: str, model: str = "whisper-1") -> str:
    """
    Transcribe speech audio using OpenAI Whisper API.
    """
    client = OpenAI(api_key=get_openai_api_key())
    with open(audio_path, "rb") as audio_file:
        result = client.audio.transcriptions.create(
            model=model,
            file=audio_file,
        )
    return result.text.strip()
