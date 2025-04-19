import pyaudio
import threading
import time
import queue
import os
import wave
import io
from typing import Callable
from pydub import AudioSegment
from dotenv import load_dotenv
from soarm100.agentic.tts.elevenlabs_model import TTSElevenLabsModel
from soarm100.agentic.llm.agent import AgentSOARM100
from soarm100.agentic.robot import SOARM100AgenticPolicy
from soarm100.agentic.utils import init_config
from soarm100.agentic.stt.whisper_model import STTWhisperModel

load_dotenv()


class AudioController:

    '''
    This class is used to control the audio input and output audio 
    for recording and playback.
    '''

    def __init__(self):
        self.p = pyaudio.PyAudio()

        self.input_stream = None  
        self.audio_queue = queue.Queue()
        self.stop_audio_playback = threading.Event()
        self.output_stream = None
        
        self.CHUNK_SIZE = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        
        self.audio_playback_thread = threading.Thread(
            target=self.play_audio_thread, 
            daemon=True
        )
        self.audio_playback_thread.start()

    def start_input_stream(self):
        if self.input_stream is None:
            self.input_stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=1024
            )
    
    def stop_input_stream(self):
        if self.input_stream is not None:
            self.input_stream.stop_stream()
            self.input_stream.close()
            self.input_stream = None

    def play_audio_thread(self):
        while not self.stop_audio_playback.is_set():
            try:
                audio_data = self.audio_queue.get(timeout=0.5)
                
                if audio_data is None:
                    continue
                
                if self.output_stream is not None:
                    self.output_stream.stop_stream()
                    self.output_stream.close()
                
                self.output_stream = self.p.open(
                    format=self.FORMAT,
                    channels=self.CHANNELS,
                    rate=self.RATE,
                    output=True
                )
                
                self.output_stream.write(audio_data)
                
                self.output_stream.stop_stream()
                self.output_stream.close()
                self.output_stream = None
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in audio playback thread: {e}")
                time.sleep(0.1)

    def cleanup(self):
        self.stop_audio_playback.set()
        if self.audio_playback_thread.is_alive():
            self.audio_playback_thread.join(timeout=1)
            
        if self.input_stream is not None:
            self.input_stream.stop_stream()
            self.input_stream.close()
            self.input_stream = None
            
        if self.output_stream is not None:
            self.output_stream.stop_stream()
            self.output_stream.close()
            self.output_stream = None
            
        self.p.terminate()


class RecordingController:
    
    def __init__(self, audio_controller: AudioController, log_callback: Callable):
        self.audio_controller = audio_controller
        self.log_callback = log_callback
        self.recording = False
        self.speech_buffer = []
        self.recording_thread = None
        self.should_stop = threading.Event()
        self.should_stop_playback = threading.Event()
        
        #cfg = init_config()
        self.stt_model    = STTWhisperModel()
        self.tts_model    = TTSElevenLabsModel()
        #self.robot_policy = SOARM100AgenticPolicy(cfg)
        #self.agent        = AgentSOARM100(self.robot_policy)

        #self.robot_thread = threading.Thread(target=self.robot_policy._run, daemon=True)
        #self.robot_thread.start()

        self.response_buffer = queue.Queue()
        self.response_thread = threading.Thread(target=self._process_response_loop, daemon=True)
        self.response_thread.start()
        self.current_audio_playing = threading.Event()
        
        self.accumulated_audio_buffer = io.BytesIO()
        self.accumulated_audio_lock = threading.Lock()

    def start_recording(self):
        if self.recording_thread and self.recording_thread.is_alive():
            self.log_callback("Recording already in progress")
            return

        self.should_stop_playback.set()
        
        try:
            self.audio_controller.start_input_stream()
            self.recording = True
            self.should_stop.clear()
            self.should_stop_playback.clear()
            self.speech_buffer = []
            self.log_callback("Recording started...")
            
            self.recording_thread = threading.Thread(target=self._recording_loop)
            self.recording_thread.start()
        except Exception as e:
            self.recording = False
            raise e

    def stop_recording(self):
        if not self.recording:
            return
        
        try:
            self.recording = False
            self.should_stop.set()
            self.should_stop_playback.clear()
            self.log_callback("Recording stopped.")
            
            if self.recording_thread:
                self.recording_thread.join(timeout=2) 
            
            self.audio_controller.stop_input_stream()
            
            if self.speech_buffer:
                self._process_recording()
        except Exception as e:
            self.log_callback(f"Error stopping recording: {e}")
            raise
        finally:
            self.recording = False 

    def _recording_loop(self):
        '''
        This functions is used to record the audio from the microphone. 
        Reads 480 frames from the audio stream and appends them to the speech buffer.
        '''
        while not self.should_stop.is_set() and self.recording:
            frame = self.audio_controller.input_stream.read(480, exception_on_overflow=False)
            self.speech_buffer.append(frame)
            time.sleep(0.001)

    def _process_recording(self):
        try:
            audio_data = b"".join(self.speech_buffer)
            temp_wav = "temp_recording.wav"
            with wave.open(temp_wav, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(audio_data)

            user_text = self.stt_model.transcribe(temp_wav)
            self.log_callback(f"[USER]: {user_text}")
            
            self.should_stop_playback.clear()
            
            with self.accumulated_audio_lock:
                self.accumulated_audio_buffer = io.BytesIO()

            self._process_response(user_text)

        except Exception as e:
            self.log_callback(f"Error: {e}")
        finally:
            if os.path.exists(temp_wav):
                os.remove(temp_wav)

    def _process_response(self, user_text: str):
        '''
        Generates the chunks of the agent's response and sends them to the buffer
        for processing in streaming
        '''
        response_text = ""
        text_buffer = ""
        try:
            for chunk in self.agent.run(user_text):
                if self.should_stop_playback.is_set():
                    break
                response_text += chunk
                text_buffer += chunk
                text_buffer = self._send_to_buffer(text_buffer)
            
            self.log_callback(f"[AGENT]: {response_text}")
            
            if text_buffer.strip():
                self.response_buffer.put(text_buffer.strip())
            
            self.response_buffer.put(None)
        except Exception as e:
            self.log_callback(f"Error in process response: {e}")

    def _process_response_loop(self):
        while True:
            try:
                response = self.response_buffer.get(timeout=0.1)
                if response is None:
                    continue

                if self.should_stop_playback.is_set():
                    continue

                while self.current_audio_playing.is_set() and not self.should_stop_playback.is_set():
                    time.sleep(0.1)

                if self.should_stop_playback.is_set():
                    continue

                self.current_audio_playing.set()
                try:
                    temp_buffer = io.BytesIO()
                    
                    for audio_chunk in self.tts_model.generate_audio_stream(response):
                        if self.should_stop_playback.is_set():
                            break
                        temp_buffer.write(audio_chunk)
                    
                    if temp_buffer.tell() > 0:
                        temp_buffer.seek(0)
                        try:
                            audio_segment = AudioSegment.from_file(temp_buffer, format="mp3")
                            
                            audio_segment = audio_segment.set_frame_rate(44100)
                            audio_segment = audio_segment.set_channels(1)
                            
                            pcm_data = audio_segment.raw_data
                            self.audio_controller.audio_queue.put(pcm_data)
                            
                        except Exception as e:
                            self.log_callback(f"Error processing audio with pydub: {e}")
                except Exception as e:
                    self.log_callback(f"Error generating audio: {e}")
                finally:
                    self.current_audio_playing.clear()
            except queue.Empty:
                continue
            except Exception as e:
                self.log_callback(f"Error in response loop: {e}")
                continue

    def _send_to_buffer(self, text_buffer: str) -> str:
        """
        Verify if there are complete sentences in the buffer and send them to the queue 
        if they meet the minimum size. Returns the remaining text that has not been sent yet.
        """
        min_length = 1
        punctuation_marks = ['.', '!', '?']
        last_valid_index = -1 

        for i, char in enumerate(text_buffer):
            if char in punctuation_marks:
                if i + 1 == len(text_buffer) or text_buffer[i + 1] == " ":
                    last_valid_index = i

        if last_valid_index != -1:
            text_to_send = text_buffer[:last_valid_index + 1].strip()
            if len(text_to_send) >= min_length:
                # Send the text to the response buffer
                print("Sending to buffer: ", text_to_send)
                self.response_buffer.put(text_to_send)
                return text_buffer[last_valid_index + 1:].lstrip()  

        return text_buffer

    def cleanup(self):
        if self.recording:
            self.stop_recording()
        
        self.should_stop_playback.set()
        self.should_stop.set()
            
        if hasattr(self, "robot_policy"):
            self.robot_policy.running = False
            if hasattr(self, "robot_thread") and self.robot_thread.is_alive():
                self.robot_thread.join(timeout=1)