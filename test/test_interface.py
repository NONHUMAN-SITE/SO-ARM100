import os
import pyaudio
import openai
import threading
import time
import numpy as np
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv
import io
import wave
from pydub import AudioSegment
import multiprocessing
import queue
import flet as ft
from datetime import datetime

load_dotenv()

print(os.getenv("ELEVENLABS_API_KEY"))

# Estados del sistema
STATE_IDLE = 0
STATE_RECORDING = 1
STATE_PROCESSING = 2
STATE_SPEAKING = 3

current_state = STATE_IDLE
user_speech_buffer = []
speech_start_time = None
speech_end_time = None

# Eventos / flags para manejar interrupciones y reproducción
interrupt_flag = threading.Event()          # Indica que se detectó voz durante TTS
stop_audio_playback = threading.Event()       # Para detener la reproducción
# Flag para detener el TTS en el proceso worker
stop_tts_flag = multiprocessing.Event()

# Variable global para guardar la referencia al proceso de TTS (inicialmente None)
tts_process = None

# Inicializar entrada de audio (micrófono)
p = pyaudio.PyAudio()
input_stream = p.open(format=pyaudio.paInt16,
                      channels=1,
                      rate=16000,
                      input=True,
                      frames_per_buffer=1024)

# Inicializar ElevenLabs
client_elevenlabs = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

# Cola global para reproducir audio (multiprocessing.Queue entre proceso TTS y thread de reproducción)
audio_queue = multiprocessing.Queue()

# Modificar los umbrales de detección de voz
VAD_SPEECH_TRIGGER = 5    # Número de frames de voz consecutivos para iniciar grabación
VAD_SILENCE_TRIGGER = 30   # Número de frames de silencio consecutivos para detener grabación
MIN_RECORDING_DURATION = 0.5  # Segundos mínimos de grabación antes de procesar

# Agregar variable global para la función de logging
log_callback = None

def set_log_callback(callback):
    global log_callback
    log_callback = callback

def log_message(message):
    global log_callback
    if log_callback:
        log_callback(message)
    print(message)  # Mantener print para debugging

class OpenAIWhisperModel:
    def __init__(self):
        self.model = openai.OpenAI()

    def transcribe(self, audio_data, fp16=False):
        # Crear archivo WAV en memoria a partir del audio capturado
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)      # mono
            wav_file.setsampwidth(2)        # 16 bits por muestra
            wav_file.setframerate(16000)    # 16 kHz
            wav_file.writeframes(audio_data)
        wav_buffer.seek(0)
        # Enviar el archivo WAV a la API de Whisper
        response = self.model.audio.transcriptions.create(
            model="whisper-1",
            file=("audio.wav", wav_buffer, "audio/wav")
        )
        return response

model = OpenAIWhisperModel()

def play_audio_thread():
    """
    Thread encargado de reproducir los chunks de audio que se depositan en la cola.
    Si se activa la bandera de detener reproducción, se vacía la cola y se sale.
    """
    p_out = pyaudio.PyAudio()
    stream_out = p_out.open(format=pyaudio.paFloat32,
                            channels=1,
                            rate=44100,
                            output=True)
    try:
        while True:
            if stop_audio_playback.is_set():
                # Vaciar la cola para no reproducir audio pendiente.
                while not audio_queue.empty():
                    try:
                        audio_queue.get_nowait()
                    except queue.Empty:
                        break
                break
            try:
                chunk = audio_queue.get(timeout=0.01)
                if stop_audio_playback.is_set():
                    break
                stream_out.write(chunk.tobytes())
            except queue.Empty:
                time.sleep(0.001)
    finally:
        stream_out.stop_stream()
        stream_out.close()
        p_out.terminate()

def tts_worker(text, audio_queue, stop_tts_flag, api_key):
    """
    Función que se ejecuta en un proceso separado para obtener el stream del TTS
    y enviar chunks de audio a la cola inmediatamente.
    """
    try:
        # Crear nuevo cliente ElevenLabs en el proceso worker
        client = ElevenLabs(api_key=api_key)
        
        # Obtener el stream de audio directamente de ElevenLabs
        audio_stream = client.text_to_speech.convert_as_stream(
            text=text,
            voice_id="TX3LPaxmHKxFdv7VOQHJ",
            model_id="eleven_turbo_v2_5"
        )

        # Procesar el stream en chunks pequeños para reproducción inmediata
        audio_buffer = io.BytesIO()
        for chunk in audio_stream:
            if stop_tts_flag.is_set():
                return
                
            # Acumular suficientes datos para procesar
            audio_buffer.write(chunk)
            
            # Cuando tengamos suficientes datos, convertir y enviar
            if audio_buffer.tell() >= 4096:  # Procesar cada 4KB
                audio_buffer.seek(0)
                try:
                    # Convertir MP3 a PCM
                    audio_segment = AudioSegment.from_mp3(audio_buffer)
                    samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32) / 32768.0
                    
                    # Enviar chunks pequeños a la cola
                    chunk_size = 1024  # Tamaño más pequeño para menor latencia
                    for i in range(0, len(samples), chunk_size):
                        if stop_tts_flag.is_set():
                            return
                        chunk = samples[i:i + chunk_size]
                        audio_queue.put(chunk)
                except Exception as e:
                    print(f"Error processing audio chunk: {e}")
                    
                # Limpiar el buffer y mantener cualquier dato restante
                remaining_data = audio_buffer.read()
                audio_buffer = io.BytesIO()
                audio_buffer.write(remaining_data)

        # Procesar cualquier dato restante
        if audio_buffer.tell() > 0:
            audio_buffer.seek(0)
            try:
                audio_segment = AudioSegment.from_mp3(audio_buffer)
                samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32) / 32768.0
                for i in range(0, len(samples), 1024):
                    if stop_tts_flag.is_set():
                        return
                    chunk = samples[i:i + 1024]
                    audio_queue.put(chunk)
            except Exception as e:
                print(f"Error processing final chunk: {e}")

    except Exception as e:
        print(f"Error in tts_worker: {e}")

def chatbot(user_text):
    # Respuesta genérica para extender el audio
    return "You said: " + user_text + ". Esto es una respuesta larga para que puedas interrumpirme. " * 5

# Iniciar thread de reproducción de audio
audio_playback_thread = threading.Thread(target=play_audio_thread, daemon=True)
audio_playback_thread.start()

class AudioController:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.input_stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1024
        )
        self.audio_queue = multiprocessing.Queue()
        self.stop_audio_playback = threading.Event()
        self.stop_tts_flag = multiprocessing.Event()
        self.tts_process = None
        
        # Asegurarse de que la clave API se cargue correctamente
        load_dotenv()
        elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        if not elevenlabs_api_key:
            raise ValueError("ELEVENLABS_API_KEY not found in environment variables")
        self.client_elevenlabs = ElevenLabs(api_key=elevenlabs_api_key)

        # Iniciar thread de reproducción
        self.audio_playback_thread = threading.Thread(
            target=self.play_audio_thread, 
            daemon=True
        )
        self.audio_playback_thread.start()

    def play_audio_thread(self):
        p_out = pyaudio.PyAudio()
        stream_out = p_out.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=44100,
            output=True
        )
        try:
            while True:
                if self.stop_audio_playback.is_set():
                    while not self.audio_queue.empty():
                        try:
                            self.audio_queue.get_nowait()
                        except queue.Empty:
                            break
                    break
                try:
                    chunk = self.audio_queue.get(timeout=0.01)
                    if self.stop_audio_playback.is_set():
                        break
                    stream_out.write(chunk.tobytes())
                except queue.Empty:
                    time.sleep(0.001)
        finally:
            stream_out.stop_stream()
            stream_out.close()
            p_out.terminate()

    def cleanup(self):
        self.input_stream.stop_stream()
        self.input_stream.close()
        self.p.terminate()

class RecordingController:
    def __init__(self, audio_controller, log_callback):
        self.audio_controller = audio_controller
        self.log_callback = log_callback
        self.recording = False
        self.speech_buffer = []
        self.recording_thread = None
        self.should_stop = threading.Event()
        self.whisper_model = OpenAIWhisperModel()
        
        # Asegurarse de que las claves API estén disponibles
        load_dotenv()
        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        if not self.elevenlabs_api_key:
            raise ValueError("ELEVENLABS_API_KEY not found in environment variables")
        
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        self.openai_client = openai.OpenAI(api_key=openai_api_key)

    def start_recording(self):
        if self.recording_thread and self.recording_thread.is_alive():
            return
            
        self.recording = True
        self.should_stop.clear()
        self.speech_buffer = []
        self.log_callback("Recording started...")
        
        self.recording_thread = threading.Thread(target=self._recording_loop)
        self.recording_thread.start()

    def stop_recording(self):
        if not self.recording:
            return
            
        self.recording = False
        self.should_stop.set()
        self.log_callback("Recording stopped.")
        
        if self.recording_thread:
            self.recording_thread.join()
        
        self._process_recording()

    def _recording_loop(self):
        while not self.should_stop.is_set():
            frame = self.audio_controller.input_stream.read(480, exception_on_overflow=False)
            self.speech_buffer.append(frame)
            time.sleep(0.001)

    def _process_recording(self):
        try:
            audio_data = b''.join(self.speech_buffer)
            self.log_callback("Transcribing audio...")
            
            # Transcribir audio
            transcription = self.whisper_model.transcribe(audio_data)
            user_text = transcription.text
            self.log_callback(f"Transcription: {user_text}")

            # Generar respuesta del chatbot de manera asíncrona
            self.log_callback("Generating response...")
            #response = self._generate_chat_response(user_text)
            response = self.chatbot(user_text)
            self.log_callback("Starting audio streaming...")
            
            # Preparar reproducción de audio
            self.audio_controller.stop_tts_flag.clear()
            self.audio_controller.stop_audio_playback.clear()
            
            # Limpiar cola de audio existente
            while not self.audio_controller.audio_queue.empty():
                try:
                    self.audio_controller.audio_queue.get_nowait()
                except queue.Empty:
                    break

            # Iniciar proceso de TTS con streaming
            self.audio_controller.tts_process = multiprocessing.Process(
                target=tts_worker,
                args=(
                    response,
                    self.audio_controller.audio_queue,
                    self.audio_controller.stop_tts_flag,
                    self.elevenlabs_api_key  # Pasar la clave API directamente
                )
            )
            self.audio_controller.tts_process.start()

        except Exception as e:
            self.log_callback(f"Error: {e}")

    def _generate_chat_response(self, user_text):
        """
        Genera una respuesta usando el API de ChatGPT
        """
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_text}
                ],
                temperature=0.7,
                max_tokens=150
            )
            return response.choices[0].message.content
        except Exception as e:
            self.log_callback(f"Error generating chat response: {e}")
            return f"I apologize, but I encountered an error: {str(e)}"

    def chatbot(self, user_text):
        return "You said: " + user_text + ". This is a long response so you can interrupt me. " * 2

    def interrupt_tts(self):
        self.log_callback("Interrupting response...")
        self.audio_controller.stop_tts_flag.set()
        self.audio_controller.stop_audio_playback.set()
        
        if self.audio_controller.tts_process and self.audio_controller.tts_process.is_alive():
            self.audio_controller.tts_process.terminate()
            self.audio_controller.tts_process.join()
        
        while not self.audio_controller.audio_queue.empty():
            try: 
                self.audio_controller.audio_queue.get_nowait()
            except queue.Empty: 
                break

class Interface:
    def __init__(self, page: ft.Page):
        self.page = page
        self.setup_page()
        self.audio_controller = AudioController()
        self.recording_controller = RecordingController(
            self.audio_controller,
            self.add_log
        )
        self.setup_controls()
        self.add_log("System initialized...")

    def setup_page(self):
        self.page.title = "AGENT SOARM100"
        self.page.bgcolor = ft.colors.WHITE
        self.page.padding = 20
        self.page.theme_mode = ft.ThemeMode.LIGHT
        
        font_file = "PressStart2P-Regular.ttf"
        if not os.path.exists(font_file):
            print(f"Warning: Font file '{font_file}' not found.")
        self.page.fonts = {"PressStart2P": font_file}

    def setup_controls(self):
        self.theme_toggle = ft.IconButton(
            icon=ft.icons.DARK_MODE,
            icon_color=ft.colors.BLACK,
            icon_size=30,
            tooltip="Toggle theme",
            on_click=self.toggle_theme,
        )

        self.title = ft.Text(
            "AGENT SOARM100",
            size=40,
            weight=ft.FontWeight.BOLD,
            font_family="Courier",
            color=ft.colors.BLACK,
            text_align=ft.TextAlign.CENTER,
        )

        self.log_area = ft.ListView(
            expand=True,
            spacing=10,
            auto_scroll=True,
            padding=10,
        )

        self.log_container = ft.Container(
            content=self.log_area,
            width=800,
            height=400,
            border=ft.border.all(2, ft.colors.BLACK),
            bgcolor=ft.colors.WHITE,
            padding=10,
        )

        self.mic_button = ft.IconButton(
            icon=ft.icons.RADIO_BUTTON_ON,
            icon_color=ft.colors.BLACK,
            icon_size=100,
            tooltip="Start recording",
            style=ft.ButtonStyle(
                shape=ft.CircleBorder(),
                side=ft.BorderSide(2, ft.colors.BLACK),
            ),
            on_click=self.toggle_recording,
        )

        self.nonhuman_label = ft.Text(
            "NONHUMAN",
            size=14,
            font_family="PressStart2P",
            color=ft.colors.BLACK,
            text_align=ft.TextAlign.CENTER,
        )

        self.page.add(
            ft.Row([
                ft.Container(expand=True),
                self.theme_toggle
            ]),
            ft.Column([
                ft.Container(content=self.title, alignment=ft.alignment.center),
                ft.Container(content=self.log_container, alignment=ft.alignment.center, expand=True),
                ft.Container(content=self.mic_button, alignment=ft.alignment.center, margin=ft.margin.only(bottom=20)),
                ft.Container(content=self.nonhuman_label, alignment=ft.alignment.center),
            ], alignment=ft.MainAxisAlignment.CENTER, expand=True, spacing=20)
        )

    def toggle_theme(self, e):
        self.page.theme_mode = (
            ft.ThemeMode.LIGHT if self.page.theme_mode == ft.ThemeMode.DARK else ft.ThemeMode.DARK
        )
        is_dark = self.page.theme_mode == ft.ThemeMode.DARK
        
        self.page.bgcolor = ft.colors.BLACK if is_dark else ft.colors.WHITE
        text_color = ft.colors.WHITE if is_dark else ft.colors.BLACK
        
        self.title.color = text_color
        self.log_container.bgcolor = ft.colors.BLACK if is_dark else ft.colors.WHITE
        self.log_container.border = ft.border.all(2, text_color)
        
        for log_message in self.log_area.controls:
            log_message.color = text_color
            
        self.mic_button.icon_color = text_color
        self.mic_button.style.side = ft.BorderSide(2, text_color)
        self.nonhuman_label.color = text_color
        self.theme_toggle.icon = ft.icons.LIGHT_MODE if is_dark else ft.icons.DARK_MODE
        self.theme_toggle.icon_color = text_color
        
        self.page.update()

    def toggle_recording(self, e):
        if self.recording_controller.recording:
            self.recording_controller.stop_recording()
            self.mic_button.icon = ft.icons.RADIO_BUTTON_OFF
            self.mic_button.tooltip = "Start recording"
        else:
            self.recording_controller.start_recording()
            self.mic_button.icon = ft.icons.STOP_CIRCLE
            self.mic_button.tooltip = "Stop recording"
        self.page.update()

    def add_log(self, message):
        timestamp = datetime.now().strftime("[%H:%M:%S]")
        self.log_area.controls.append(
            ft.Text(
                f"{timestamp} {message}",
                color=ft.colors.BLACK if self.page.theme_mode == ft.ThemeMode.LIGHT else ft.colors.WHITE,
                font_family="Courier",
                size=16,
            )
        )
        self.page.update()

def main(page: ft.Page):
    interface = Interface(page)

if __name__ == "__main__":
    ft.app(target=main)