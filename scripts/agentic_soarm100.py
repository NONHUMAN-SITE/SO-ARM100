import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import flet as ft
from dotenv import load_dotenv
from datetime import datetime
from soarm100.agentic.record import AudioController, RecordingController

load_dotenv()

os.environ["ELEVENLABS_API_KEY"] = os.getenv("ELEVENLABS_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


STATE_IDLE = 0
STATE_RECORDING = 1
STATE_PROCESSING = 2
STATE_SPEAKING = 3


class Interface:
    
    def __init__(self, page: ft.Page):
        self.page = page
        self.setup_page()
        self.setup_controls()

        self.audio_controller = AudioController()
        self.recording_controller = RecordingController(
            self.audio_controller,
            self.add_log
        )
        self.add_log("System initialized...")

    def __del__(self):
        if hasattr(self, "recording_controller"):
            self.recording_controller.cleanup()
        if hasattr(self, "audio_controller"):
            self.audio_controller.cleanup()

    def setup_page(self):
        self.page.title = "AGENT SOARM100"
        self.page.bgcolor = ft.colors.WHITE
        self.page.padding = 20
        self.page.theme_mode = ft.ThemeMode.LIGHT
        
        font_file = "../fonts/PressStart2P-Regular.ttf"
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
