import os
from elevenlabs import ElevenLabs
from elevenlabs.client import VoiceSettings

class TTSElevenLabsModel:
    def __init__(self):
        self.client = ElevenLabs(
            api_key=os.environ["ELEVENLABS_API_KEY"],
        )

    def convert(self, text:str, output_path:str):
        
        #voice_settings = VoiceSettings(
        #    style=0.5,
        #    use_speaker_boost=True,
        #    stability=0.5,
        #    similarity_boost=0.5
        #)

        audio = self.client.text_to_speech.convert(
            voice_id="TX3LPaxmHKxFdv7VOQHJ",
            output_format="mp3_44100_128",
            text=text,
            model_id="eleven_multilingual_v2",
            #voice_settings=voice_settings
        )
        
        with open(output_path, 'wb') as f:
            for chunk in audio:
                f.write(chunk)

        return output_path