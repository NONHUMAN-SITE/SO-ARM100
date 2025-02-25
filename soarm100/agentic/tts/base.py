from typing import Iterator
from abc import ABC, abstractmethod

class TTSBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def generate_audio(self, text: str) -> Iterator[bytes]:
        '''
        Generates audio from text.
        '''
        pass
    
    @abstractmethod
    def generate_audio_stream(self, text: str) -> Iterator[bytes]:
        '''
        Generates audio from text in a stream.
        '''
        pass
