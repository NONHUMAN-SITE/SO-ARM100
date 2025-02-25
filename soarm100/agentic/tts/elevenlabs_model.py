import os
from elevenlabs import ElevenLabs
from typing import Iterator
from soarm100.agentic.tts.base import TTSBase

class ElevenLabsModelConfig:
    '''
    You can change the voice_id to the one you want to use.
    Check the documentation for more information:
    https://elevenlabs.io/docs/api-reference/text-to-speech/convert
    '''
    def __init__(self):
        self.voice_id = "TX3LPaxmHKxFdv7VOQHJ"
        self.output_format = "mp3_44100_128"
        self.model_id = "eleven_flash_v2_5"

class TTSElevenLabsModel(TTSBase):
    def __init__(self):
        self.client = ElevenLabs(
            api_key=os.getenv("ELEVENLABS_API_KEY"),
        )
        self.config = ElevenLabsModelConfig()

    def generate_audio_stream(self, text: str):
        if not text or text.isspace():
            return []
            
        try:
            audio_stream = self.client.text_to_speech.convert_as_stream(
                text=text,
                voice_id=self.config.voice_id,
                output_format=self.config.output_format,
                model_id=self.config.model_id,
                language_code="es"
            )
            
            for chunk in audio_stream:
                yield chunk
        except Exception as e:
            print(f"Error al generar audio streaming: {e}")
            yield b""

    def generate_audio(self, text: str) -> bytes:
        if not text or text.isspace():
            return b""
            
        try:
            audio = self.client.text_to_speech.convert(
                text=text,
                voice_id=self.config.voice_id,
                output_format=self.config.output_format,
                model_id=self.config.model_id,
            )
            return audio
        except Exception as e:
            print(f"Error al generar audio: {e}")
            return b""