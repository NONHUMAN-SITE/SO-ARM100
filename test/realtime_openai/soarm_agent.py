import asyncio
import websockets
import json
import pyaudio
import base64
import logging
import os
import ssl
import threading
import subprocess
from datetime import datetime
from colorama import init, Fore, Style

from dotenv import load_dotenv
load_dotenv()

# Initialize colorama for cross-platform colored output
init()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TOPIC = "Rubber duck"

INSTRUCTIONS = """
You are a robot called SOARM100. You were created by the NONHUMAN group.
Your mission is to obey the instructions you are given.
Your personality is that of a charismatic robot. The tone of your responses should be futuristic.
"""

KEYBOARD_COMMANDS = """
q: Quit
t: Send text message
a: Send audio message
"""

def print_banner():
    banner = f"""
{Fore.CYAN}
╔═══════════════════════════════════════════╗
║                NONHUMAN                   ║
║                                           ║
║           S O A R M - 1 0 0               ║
╚═══════════════════════════════════════════╝
{Style.RESET_ALL}"""
    print(banner)

def print_options():
    options = f"""
{Fore.GREEN}Choose your interaction mode:{Style.RESET_ALL}
{Fore.YELLOW}1{Style.RESET_ALL}: Voice Chat (Talk with SOARM)
{Fore.YELLOW}2{Style.RESET_ALL}: Text Chat (Type with SOARM)
{Fore.YELLOW}3{Style.RESET_ALL}: Both (Voice and Text)
{Fore.YELLOW}q{Style.RESET_ALL}: Quit

Enter your choice: """
    return input(options)

def print_message(role, message, timestamp=True):
    current_time = datetime.now().strftime("%H:%M:%S")
    if role.lower() == "soarm":
        print(f"{Fore.CYAN}[{current_time}] SOARM{Style.RESET_ALL}: {message}")
    else:
        print(f"{Fore.GREEN}[{current_time}] You{Style.RESET_ALL}: {message}")

def print_commands():
    commands = f"""
{Fore.YELLOW}Available Commands:{Style.RESET_ALL}
/quit   - Exit the conversation
/mode   - Change interaction mode
/clear  - Clear the screen
/help   - Show this help message
"""
    print(commands)

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


class RealtimeClient:
    """
    Client for interacting with the OpenAI Realtime API via WebSocket.
    """
    def __init__(self, instructions, voice="alloy"):
        # WebSocket Configuration
        self.url = "wss://api.openai.com/v1/realtime"
        self.model = "gpt-4o-realtime-preview-2024-10-01"
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.ws = None
        self.audio_handler = AudioHandler()
        self.mode = None  # Will be set based on user choice
        
        # SSL Configuration
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE
        
        self.audio_buffer = b''
        self.instructions = instructions
        self.voice = voice

        # VAD mode
        self.VAD_turn_detection = True
        self.VAD_config = {
            "type": "server_vad",
            "threshold": 0.5,
            "prefix_padding_ms": 300,
            "silence_duration_ms": 600
        }

        self.session_config = {
            "modalities": ["audio", "text"],
            "instructions": self.instructions,
            "voice": self.voice,
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "turn_detection": self.VAD_config if self.VAD_turn_detection else None,
            "input_audio_transcription": {
                "model": "whisper-1"
            },
            "temperature": 0.6,
            "tool_choice": "auto",
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get current weather for a specified city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "The name of the city for which to fetch the weather."
                            }
                        },
                        "required": ["city"]
                    }
                },
                {
                    "type": "function",
                    "name": "write_notepad",
                    "description": "Open a text editor and write the time, for example, 2024-10-29 16:19. Then, write the content, which should include my questions along with your answers.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "The content consists of my questions along with the answers you provide."
                            },
                            "date": {
                                "type": "string",
                                "description": "the time, for example, 2024-10-29 16:19. "
                            }
                        },
                        "required": ["content", "date"]
                    }
                }
            ]
        }

    def set_mode(self, mode):
        """Set the interaction mode."""
        self.mode = mode
        if mode in ['1', '3']:  # Modes that include voice
            self.session_config["modalities"] = ["audio", "text"]
        else:  # Text-only mode
            self.session_config["modalities"] = ["text"]

    async def handle_command(self, command):
        """Handle special commands."""
        if command == "/quit":
            return True
        elif command == "/mode":
            new_mode = print_options()
            self.set_mode(new_mode)
        elif command == "/clear":
            os.system('cls' if os.name == 'nt' else 'clear')
            print_banner()
        elif command == "/help":
            print_commands()
        return False

    async def handle_event(self, event):
        """Handle incoming events from the WebSocket server."""
        event_type = event.get("type")
        logger.debug(f"Received event type: {event_type}")

        if event_type == "error":
            logger.error(f"Error event received: {event['error']['message']}")
        elif event_type == "response.text.delta":
            print(f"{Fore.CYAN}{event['delta']}{Style.RESET_ALL}", end="", flush=True)
        elif event_type == "response.audio.delta":
            if self.mode in ['1', '3']:  # Only handle audio in voice modes
                audio_data = base64.b64decode(event["delta"])
                self.audio_buffer += audio_data
                logger.debug("Audio data appended to buffer")
        elif event_type == "response.audio.done":
            if self.mode in ['1', '3'] and self.audio_buffer:
                self.audio_handler.play_audio(self.audio_buffer)
                logger.info("Done playing audio response")
                self.audio_buffer = b''
            else:
                logger.warning("No audio data to play")
        elif event_type == "response.done":
            print()  # New line after response
            logger.debug("Response generation completed")
        elif event_type == "conversation.item.created":
            logger.debug(f"Conversation item created: {event.get('item')}")
        elif event_type == "input_audio_buffer.speech_started":
            logger.debug("Speech started detected by server VAD")
        elif event_type == "input_audio_buffer.speech_stopped":
            logger.debug("Speech stopped detected by server VAD")
        elif event_type == "response.function_call_arguments.done":
            logger.debug("Function call received")
            await self.handle_function_call(event)
        else:
            logger.debug(f"Unhandled event type: {event_type}")

    async def connect(self):
        """
        Connect to the WebSocket server.
        """
        logger.info(f"Connecting to WebSocket: {self.url}")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1"
        }
        
        # NEEDS websockets version < 14.0
        self.ws = await websockets.connect(
            f"{self.url}?model={self.model}",
            extra_headers=headers,
            ssl=self.ssl_context
        )
        logger.info("Successfully connected to OpenAI Realtime API")

        # Configure session
        await self.send_event(
            {
                "type": "session.update",
                "session": self.session_config
            }
        )
        logger.info("Session set up")

        # Send a response.create event to initiate the conversation
        await self.send_event({"type": "response.create"})
        logger.debug("Sent response.create to initiate conversation")

    async def send_event(self, event):
        """
        Send an event to the WebSocket server.
        
        :param event: Event data to send (from the user)
        """
        await self.ws.send(json.dumps(event))
        logger.debug(f"Event sent - type: {event['type']}")

    async def receive_events(self):
        """
        Continuously receive events from the WebSocket server.
        """
        try:
            async for message in self.ws:
                event = json.loads(message)
                await self.handle_event(event)
        except websockets.ConnectionClosed as e:
            logger.error(f"WebSocket connection closed: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")

    async def send_text(self, text):
        """
        Send a text message to the WebSocket server.
        
        :param text: Text message to send.
        """
        logger.info(f"Sending text message: {text}")
        event = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{
                    "type": "input_text",
                    "text": text
                }]
            }
        }
        await self.send_event(event)
        await self.send_event({"type": "response.create"})
        logger.debug(f"Sent text: {text}")

    async def send_audio(self):
        """
        Record and send audio using server-side turn detection
        """
        logger.debug("Starting audio recording for user input")
        self.audio_handler.start_recording()
        
        try:
            while True:
                chunk = self.audio_handler.record_chunk()
                if chunk:
                    # Encode and send audio chunk
                    base64_chunk = base64.b64encode(chunk).decode('utf-8')
                    await self.send_event({
                        "type": "input_audio_buffer.append",
                        "audio": base64_chunk
                    })
                    await asyncio.sleep(0.01)
                else:
                    break

        except Exception as e:
            logger.error(f"Error during audio recording: {e}")
            self.audio_handler.stop_recording()
            logger.debug("Audio recording stopped")
    
        finally:
            # Stop recording even if an exception occurs
            self.audio_handler.stop_recording()
            logger.debug("Audio recording stopped")
        
        # Commit the audio buffer if VAD is disabled
        if not self.VAD_turn_detection:
            await self.send_event({"type": "input_audio_buffer.commit"})
            logger.debug("Audio buffer committed")
        
        # When in Server VAD mode, the client does not need to send this event, the server will commit the audio buffer automatically.
        # https://platform.openai.com/docs/api-reference/realtime-client-events/input_audio_buffer/commit

    async def run(self):
        """
        Main loop to handle user input and interact with the WebSocket server.
        """
        await self.connect()
        
        # Continuously listen to events in the background
        receive_task = asyncio.create_task(self.receive_events())

        try:
            while True:
                if self.mode in ['2', '3']:  # Text input modes
                    command = input(f"{Fore.GREEN}You{Style.RESET_ALL}: ")
                    if command.startswith('/'):
                        if await self.handle_command(command):
                            break
                    else:
                        print_message("you", command)
                        await self.send_text(command)
                elif self.mode == '1':  # Voice-only mode
                    print(f"{Fore.YELLOW}Press Enter to start speaking, or type a command:{Style.RESET_ALL}")
                    command = input()
                    if command.startswith('/'):
                        if await self.handle_command(command):
                            break
                    else:
                        await self.send_audio()
                await asyncio.sleep(0.1)
        except Exception as e:
            logger.error(f"An error occurred: {e}")
        finally:
            receive_task.cancel()
            await self.cleanup()

    async def cleanup(self):
        """
        Clean up resources by closing the WebSocket and audio handler.
        """
        self.audio_handler.cleanup()
        if self.ws:
            await self.ws.close()

    async def handle_function_call(self, event):
        """
        Handle function calls from the OpenAI API.
        """
        try:
            name = event.get("name", "")
            call_id = event.get("call_id", "")
            arguments = event.get("arguments", "{}")
            function_call_args = json.loads(arguments)

            if name == "write_notepad":
                logger.info(f"Starting write_notepad function, event = {event}")
                content = function_call_args.get("content", "")
                date = function_call_args.get("date", "")

                subprocess.Popen(
                    ["powershell", "-Command", f"Add-Content -Path temp.txt -Value 'date: {date}\n{content}\n\n'; notepad.exe temp.txt"])

                await self.send_function_call_result("write notepad successful.", call_id)

            elif name == "get_weather":
                city = function_call_args.get("city", "")
                if city:
                    weather_result = self.get_weather(city)
                    await self.send_function_call_result(weather_result, call_id)
                else:
                    logger.warning("City not provided for get_weather function.")
        except Exception as e:
            logger.error(f"Error parsing function call arguments: {e}")

    async def send_function_call_result(self, result, call_id):
        """
        Send the result of a function call back to the server.
        """
        result_json = {
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "output": result,
                "call_id": call_id
            }
        }

        try:
            await self.send_event(result_json)
            logger.debug(f"Sent function call result: {result_json}")

            rp_json = {
                "type": "response.create"
            }
            await self.send_event(rp_json)
            logger.debug(f"Sent response.create after function call")
        except Exception as e:
            logger.error(f"Failed to send function call result: {e}")

    def get_weather(self, city):
        """
        Simulate retrieving weather information for a given city.
        """
        return json.dumps({
            "city": city,
            "temperature": "99°C"
        })

async def main():
    # Clear screen and show banner
    os.system('cls' if os.name == 'nt' else 'clear')
    print_banner()
    print_commands()
    
    # Get initial mode
    mode = print_options()
    if mode.lower() == 'q':
        return

    client = RealtimeClient(
        instructions=INSTRUCTIONS,
        voice="alloy"
    )
    client.set_mode(mode)
    
    try:
        await client.run()
    except Exception as e:
        logger.error(f"An error occurred in main: {e}")
    finally:
        print(f"\n{Fore.YELLOW}Goodbye! Thank you for using SOARM-100{Style.RESET_ALL}")
        logger.info("Main done")

if __name__ == "__main__":
    asyncio.run(main())