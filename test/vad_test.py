import os
import pyaudio
import webrtcvad    
import openai
import requests
import threading
import time
import numpy as np
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv
import tiktoken
import io
import wave
from pydub import AudioSegment
import multiprocessing
import queue

load_dotenv()

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

# Inicializar VAD con alta sensibilidad
vad = webrtcvad.Vad(mode=3)

# Inicializar ElevenLabs
client_elevenlabs = ElevenLabs()

# Cola global para reproducir audio (multiprocessing.Queue entre proceso TTS y thread de reproducción)
audio_queue = multiprocessing.Queue()

# Modificar los umbrales de detección de voz
VAD_SPEECH_TRIGGER = 5    # Número de frames de voz consecutivos para iniciar grabación
VAD_SILENCE_TRIGGER = 30   # Número de frames de silencio consecutivos para detener grabación
MIN_RECORDING_DURATION = 0.5  # Segundos mínimos de grabación antes de procesar

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

def vad_thread():
    """
    Thread que lee el micrófono continuamente.
    Si el estado es SPEAKING y se detecta voz, activa los flags para interrumpir TTS 
    y detener su reproducción.
    También detecta el inicio y final del habla para iniciar la grabación.
    """
    global current_state, user_speech_buffer, speech_start_time, speech_end_time, tts_process
    speech_count = 0
    no_speech_count = 0
    last_interruption_time = 0  # Para evitar detecciones inmediatas post-interrupción
    
    while True:
        frame = input_stream.read(480, exception_on_overflow=False)
        current_time = time.time()
        
        try:
            is_speech = vad.is_speech(frame, 16000)
        except Exception:
            continue
        
        # Cooldown de 1 segundo post-interrupción para evitar falsos positivos
        if current_time - last_interruption_time < 1.0:
            continue
            
        # Lógica de interrupción
        if current_state == STATE_SPEAKING and is_speech:
            print("Detectada interrupción...")
            last_interruption_time = current_time
            interrupt_flag.set()
            stop_tts_flag.set()
            stop_audio_playback.set()
            
            if tts_process and tts_process.is_alive():
                tts_process.terminate()
                tts_process.join()
            
            while not audio_queue.empty():
                try: audio_queue.get_nowait()
                except queue.Empty: break
            
            # Reinicio completo del sistema
            current_state = STATE_IDLE
            user_speech_buffer = []
            speech_count = 0
            no_speech_count = 0
            interrupt_flag.clear()
            stop_tts_flag.clear()
            stop_audio_playback.clear()
        
        # Lógica principal de detección de voz
        if current_state == STATE_IDLE:
            if is_speech:
                speech_count += 1
                if speech_count >= VAD_SPEECH_TRIGGER:
                    print("Iniciando grabación...")
                    current_state = STATE_RECORDING
                    user_speech_buffer = [frame]
                    speech_start_time = time.time()
                    speech_count = 0
            else:
                speech_count = max(0, speech_count - 1)  # Decaimiento gradual

        elif current_state == STATE_RECORDING:
            user_speech_buffer.append(frame)
            
            # Verificar duración mínima antes de permitir detención
            if time.time() - speech_start_time < MIN_RECORDING_DURATION:
                continue
                
            if not is_speech:
                no_speech_count += 1
                if no_speech_count >= VAD_SILENCE_TRIGGER:
                    print("Finalizando grabación...")
                    current_state = STATE_PROCESSING
                    speech_end_time = time.time()
                    no_speech_count = 0
            else:
                no_speech_count = max(0, no_speech_count - 2)  # Tolerancia a pausas breves

def tts_worker(text, audio_queue, stop_tts_flag):
    """
    Función que se ejecuta en un proceso separado para obtener el stream del TTS
    y enviar chunks de audio a la cola.
    Se verifica el flag stop_tts_flag en cada paso para permitir la interrupción.
    """
    import io
    import numpy as np
    from pydub import AudioSegment
    try:
        audio_stream = client_elevenlabs.text_to_speech.convert_as_stream(
            text=text,
            voice_id="TX3LPaxmHKxFdv7VOQHJ",
            model_id="eleven_turbo_v2_5"
        )
        audio_buffer = io.BytesIO()
        for chunk in audio_stream:
            if stop_tts_flag.is_set():
                print("Interrumpido TTS!!! (worker)")
                return
            audio_buffer.write(chunk)
        
        if stop_tts_flag.is_set():
            print("Interrumpido TTS!!! (worker) post stream")
            return
        
        audio_buffer.seek(0)
        audio_segment = AudioSegment.from_mp3(audio_buffer)
        samples = audio_segment.get_array_of_samples()
        samples = np.array(samples).astype(np.float32) / 32768.0
        
        chunk_size = 4096
        for i in range(0, len(samples), chunk_size):
            if stop_tts_flag.is_set():
                print("Interrumpido TTS durante la reproducción!!! (worker)")
                break
            chunk = samples[i:i + chunk_size]
            audio_queue.put(chunk)
    except Exception as e:
        print("Error en tts_worker:", e)

def chatbot(user_text):
    # Respuesta genérica para extender el audio
    return "You said: " + user_text + ". Esto es una respuesta larga para que puedas interrumpirme. " * 5

# Iniciar thread de reproducción de audio
audio_playback_thread = threading.Thread(target=play_audio_thread, daemon=True)
audio_playback_thread.start()

# Iniciar thread de detección de voz (VAD)
vad_thread_obj = threading.Thread(target=vad_thread, daemon=True)
vad_thread_obj.start()

while True:
    if current_state == STATE_IDLE:
        # Esperamos a detectar voz
        pass
    elif current_state == STATE_RECORDING:
        # Continuamos acumulando audio
        pass
    elif current_state == STATE_PROCESSING:
        try:
            # Calcular duración real del audio
            duration = speech_end_time - speech_start_time
            if duration < MIN_RECORDING_DURATION:
                print("Grabación demasiado corta, descartando...")
                current_state = STATE_IDLE
                continue
                
            audio_data = b''.join(user_speech_buffer)
            transcription = model.transcribe(audio_data)
            user_text = transcription.text
            print(f"Transcripción: {user_text}")
            
            # Procesar respuesta
            response = chatbot(user_text)
            current_state = STATE_SPEAKING
            
            # Reiniciar sistema de audio
            stop_tts_flag.clear()
            stop_audio_playback.clear()
            while not audio_queue.empty():
                try: audio_queue.get_nowait()
                except queue.Empty: break
            
            # Iniciar TTS
            tts_process = multiprocessing.Process(
                target=tts_worker, 
                args=(response, audio_queue, stop_tts_flag)
            )
            tts_process.start()
            
        except Exception as e:
            print(f"Error: {e}")
            current_state = STATE_IDLE

    time.sleep(0.1)

# Al finalizar (si se llega a finalizar el bucle)
input_stream.stop()
input_stream.close()
p.terminate()