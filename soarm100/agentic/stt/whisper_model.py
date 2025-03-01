import os
from openai import OpenAI
from soarm100.agentic.stt.base import STTBase

class STTWhisperModel(STTBase):
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def transcribe(self, input_path:str) -> str:
        audio_file= open(input_path, "rb")
        transcription = self.client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file
        )
        return transcription.text


