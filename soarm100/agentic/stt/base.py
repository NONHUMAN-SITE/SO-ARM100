from abc import ABC, abstractmethod

class STTBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def transcribe(self, audio_data: bytes) -> str:
        pass
