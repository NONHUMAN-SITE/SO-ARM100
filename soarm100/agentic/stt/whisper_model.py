import os
from openai import OpenAI

class STTWhisperModel:
    def __init__(self):
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def transcribe(self, input_path:str) -> str:
        audio_file= open(input_path, "rb")
        transcription = self.client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file
        )
        return transcription.text


