import sys
sys.path.append("/home/leonardo/NONHUMAN/SO-ARM100/")

import os
import time
import pyaudio
import wave
import threading
from dotenv import load_dotenv
from soarm100.agentic.robot import SOARM100AgenticPolicy
from soarm100.agentic.utils import init_config
from soarm100.agentic.llm.agent import AgentSOARM100
from soarm100.agentic.stt.whisper_model import STTWhisperModel
from soarm100.agentic.tts.elevenlabs_model import TTSElevenLabsModel
from pydub import AudioSegment
from pydub.playback import play

load_dotenv()

os.environ["SDL_VIDEODRIVER"] = "dummy"

os.environ["AUDIO_INPUT_DATA_PATH"] = "/home/leonardo/NONHUMAN/SO-ARM100/scripts/audio/input"
os.environ["AUDIO_OUTPUT_DATA_PATH"] = "/home/leonardo/NONHUMAN/SO-ARM100/scripts/audio/output"

def ending_recording(audio, frames, stream, record_thread):

    record_thread.join()
    stream.stop_stream()
    stream.close()
    
    path_audio = os.path.join(os.environ["AUDIO_INPUT_DATA_PATH"], "grabacion.mp3")
    sound_file = wave.open(path_audio, "wb")
    sound_file.setnchannels(1)
    sound_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
    sound_file.setframerate(44100)
    sound_file.writeframes(b"".join(frames))
    sound_file.close()

def play_audio_file(audio_path):
    """
    Play audio from the given file path (supports MP3)
    """
    try:
        audio = AudioSegment.from_mp3(audio_path)
        play(audio)
    except Exception as e:
        print(f"Error playing audio: {e}")

def main_workflow(agent:AgentSOARM100,
                  tts:TTSElevenLabsModel,
                  stt:STTWhisperModel):


    audio = pyaudio.PyAudio()
    frames = []
    recording = False
    stream = None
    
    def record_audio():
        nonlocal frames, stream
        while recording:
            data = stream.read(1024)
            frames.append(data)
    
    while True:
        command = input("Presiona 'r' para grabar, 's' para detener, 'q' para salir: ").lower()
        
        if command == 'r' and not recording:
            recording = True
            try:
                os.remove(os.path.join(os.environ["AUDIO_INPUT_DATA_PATH"], "grabacion.mp3"))
            except:
                pass
            stream = audio.open(format=pyaudio.paInt16,
                              channels=1,
                              rate=44100,
                              input=True,
                              frames_per_buffer=1024)
            frames = []
            print("Grabando... (presiona 's' para detener)")
            
            record_thread = threading.Thread(target=record_audio)
            record_thread.start()
            
        elif command == 's' and recording:
            recording = False
            ending_recording(audio, frames, stream, record_thread)
            print("Ending recording...")
            start_time = time.time()
            transcription = stt.transcribe(os.path.join(os.environ["AUDIO_INPUT_DATA_PATH"], "grabacion.mp3"))
            print(f"Transcription: {transcription}")
            end_time = time.time()
            print(f"Transcription time: {end_time - start_time} seconds")
            
            start_time = time.time()
            response = agent.run(transcription)
            end_time = time.time()
            print(f"Response time: {end_time - start_time} seconds")
            
            
            start_time = time.time()
            output_path = os.path.join(os.environ["AUDIO_OUTPUT_DATA_PATH"], "output.mp3")
            tts.convert(response, output_path)
            end_time = time.time()
            print(f"TTS time: {end_time - start_time} seconds")
            
            start_time = time.time()
            play_audio_file(output_path)
            end_time = time.time()
            print(f"Play time: {end_time - start_time} seconds")
        
        elif command == 'q':
            break

    audio.terminate()

def main(args=None):
    cfg          = init_config()
    robot_policy = SOARM100AgenticPolicy(cfg)
    agent        = AgentSOARM100(robot_policy)
    tts          = TTSElevenLabsModel()
    stt          = STTWhisperModel()
    
    # Variable de control para señalizar la terminación
    running = threading.Event()
    running.set()  # Inicialmente está activo
    
    # Modificar el _run del robot para que verifique la señal de terminación
    def robot_run_wrapper():
        while running.is_set():
            robot_policy._run()
            #time.sleep(1)  # Pequeña pausa para no consumir CPU innecesariamente
    
    # Crear threads para ejecutar las funciones en paralelo
    robot_thread = threading.Thread(target=robot_run_wrapper)
    workflow_thread = threading.Thread(target=main_workflow, args=(agent, tts, stt))
    
    # Configurar los threads como daemons
    robot_thread.daemon = True
    workflow_thread.daemon = True
    
    # Iniciar los threads
    robot_thread.start()
    workflow_thread.start()
    
    try:
        # Esperar a que el thread del workflow termine
        workflow_thread.join()
    except KeyboardInterrupt:
        print("\nPrograma terminado por el usuario")
    finally:
        # Señalizar la terminación y limpiar recursos
        running.clear()  # Esto hará que robot_run_wrapper termine
        # Esperar a que el thread del robot termine
        robot_thread.join(timeout=2)  # Esperar máximo 2 segundos

if __name__ == "__main__":
    main()














