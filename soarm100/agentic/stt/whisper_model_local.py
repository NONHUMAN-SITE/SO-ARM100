import torch
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available

from base import STTBase


class STTWhisperFlashModel(STTBase):
    def __init__(self,device: str = "cuda:0"):
        super().__init__()

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-small", # select checkpoint from https://huggingface.co/openai/whisper-large-v3#model-details
            torch_dtype=torch.float16,
            device=device,
            model_kwargs={"attn_implementation": "flash_attention_2"}
            if is_flash_attn_2_available() else {"attn_implementation": "sdpa"},
        )

    def transcribe(self, audio_path: str) -> str:
        '''
        Transcribe an audio file using the Whisper model.
        '''
        print("Initializing Whisper model...")

        outputs = self.pipe(
            audio_path,
            chunk_length_s=30,
            batch_size=24,
            return_timestamps=True,
            task="transcribe",
            input_features=None,
        )

        return outputs["text"]