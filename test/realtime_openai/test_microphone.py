import pyaudio
import threading
import logging
import time
import wave
import os

logger = logging.getLogger(__name__)

class AudioHandler:
    """
    Handles audio input and output using PyAudio.
    """
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.audio_buffer = b''
        self.chunk_size = 1024  # Number of audio frames per buffer
        self.format = pyaudio.paInt16  # Audio format (16-bit PCM)
        self.channels = 1  # Mono audio
        self.rate = 24000  # Sampling rate in Hz
        self.is_recording = False

    def start_audio_stream(self):
        """
        Start the audio input stream.
        """
        self.stream = self.p.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )

    def stop_audio_stream(self):
        """
        Stop the audio input stream.
        """
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()

    def cleanup(self):
        """
        Clean up resources by stopping the stream and terminating PyAudio.
        """
        if self.stream:
            self.stop_audio_stream()
        self.p.terminate()

    def start_recording(self):
        """Start continuous recording"""
        self.is_recording = True
        self.audio_buffer = b''
        self.start_audio_stream()

    def stop_recording(self):
        """Stop recording and return the recorded audio"""
        self.is_recording = False
        self.stop_audio_stream()
        return self.audio_buffer

    def record_chunk(self):
        """Record a single chunk of audio"""
        if self.stream and self.is_recording:
            data = self.stream.read(self.chunk_size)
            self.audio_buffer += data
            return data
        return None
    
    def play_audio(self, audio_data):
        """
        Play audio data.
        
        :param audio_data: Received audio data (AI response)
        """
        def play():
            stream = self.p.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                output=True
            )
            stream.write(audio_data)
            stream.stop_stream()
            stream.close()

        logger.debug("Playing audio")
        # Use a separate thread for playback to avoid blocking
        playback_thread = threading.Thread(target=play)
        playback_thread.start()

def main():
    """
    Test microphone by recording for 10 seconds and playing back the recording.
    Also saves the recording to a WAV file.
    """
    print("Microphone Test Script")
    print("---------------------")
    
    audio = AudioHandler()
    
    print("Starting recording for 10 seconds...")
    audio.start_recording()
    
    # Record for exactly 10 seconds using chunks
    start_time = time.time()
    frames = []
    
    while time.time() - start_time < 5.0:
        if data := audio.record_chunk():
            frames.append(data)
    
    print("Recording finished.")
    recorded_data = b''.join(frames)
    audio.stop_recording()
    
    # Save to WAV file
    output_file = "microphone_test.wav"
    with wave.open(output_file, 'wb') as wf:
        wf.setnchannels(audio.channels)
        wf.setsampwidth(audio.p.get_sample_size(audio.format))
        wf.setframerate(audio.rate)
        wf.writeframes(recorded_data)
    
    print(f"\nRecording saved to {os.path.abspath(output_file)}")
    print(f"Recording length: {len(recorded_data) / (audio.channels * audio.p.get_sample_size(audio.format) * audio.rate):.2f} seconds")
    print("\nPlaying back the recording...")
    
    # Play back the recording
    audio.play_audio(recorded_data)
    
    # Wait for playback to finish
    time.sleep(len(recorded_data) / (audio.channels * audio.p.get_sample_size(audio.format) * audio.rate))
    
    audio.cleanup()
    print("\nTest completed!")

if __name__ == "__main__":
    main()