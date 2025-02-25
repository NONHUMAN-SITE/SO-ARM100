import flet as ft
import os
from datetime import datetime

def main(page: ft.Page):
    page.title = "AGENT SOARM100"
    page.bgcolor = ft.colors.WHITE
    page.padding = 20
    page.theme_mode = ft.ThemeMode.LIGHT

    # Variables para los colores del tema
    RETRO_GREEN = "#00FF00"
    
    def toggle_theme(e):
        page.theme_mode = (
            ft.ThemeMode.LIGHT if page.theme_mode == ft.ThemeMode.DARK else ft.ThemeMode.DARK
        )
        # Actualizar colores basados en el tema
        if page.theme_mode == ft.ThemeMode.DARK:
            page.bgcolor = ft.colors.BLACK
            title.color = ft.colors.WHITE
            log_container.bgcolor = ft.colors.BLACK
            log_container.border = ft.border.all(2, ft.colors.WHITE)
            # Actualizar color de todos los mensajes existentes
            for log_message in log_area.controls:
                log_message.color = ft.colors.WHITE
            mic_button.icon_color = ft.colors.WHITE
            mic_button.style.side = ft.BorderSide(2, ft.colors.WHITE)
            nonhuman_label.color = ft.colors.WHITE
            theme_toggle.icon = ft.icons.LIGHT_MODE
            theme_toggle.icon_color = ft.colors.WHITE
        else:
            page.bgcolor = ft.colors.WHITE
            title.color = ft.colors.BLACK
            log_container.bgcolor = ft.colors.WHITE
            log_container.border = ft.border.all(2, ft.colors.BLACK)
            # Actualizar color de todos los mensajes existentes
            for log_message in log_area.controls:
                log_message.color = ft.colors.BLACK
            mic_button.icon_color = ft.colors.BLACK
            mic_button.style.side = ft.BorderSide(2, ft.colors.BLACK)
            nonhuman_label.color = ft.colors.BLACK
            theme_toggle.icon = ft.icons.DARK_MODE
            theme_toggle.icon_color = ft.colors.BLACK
        page.update()

    is_recording = False
    def toggle_recording(e):
        nonlocal is_recording
        is_recording = not is_recording
        if is_recording:
            mic_button.icon = ft.icons.STOP_CIRCLE
            mic_button.tooltip = "Stop recording"
            add_log("Recording started...")
        else:
            mic_button.icon = ft.icons.MIC
            mic_button.tooltip = "Start recording"
            add_log("Recording stopped.")
        page.update()

    # Botón de toggle tema
    theme_toggle = ft.IconButton(
        icon=ft.icons.DARK_MODE,
        icon_color=ft.colors.BLACK,
        icon_size=30,
        tooltip="Toggle theme",
        on_click=toggle_theme,
    )

    # Asegúrate de que el archivo de fuente esté en el mismo directorio que este script
    font_file = "PressStart2P-Regular.ttf"
    if not os.path.exists(font_file):
        print(f"Warning: Font file '{font_file}' not found. Using default font.")

    page.fonts = {
        "PressStart2P": font_file
    }

    def add_log(message):
        timestamp = datetime.now().strftime("[%H:%M:%S]")
        log_area.controls.append(
            ft.Text(
                f"{timestamp} {message}",
                color=ft.colors.BLACK if page.theme_mode == ft.ThemeMode.LIGHT else ft.colors.WHITE,
                font_family="Courier",
                size=16,
            )
        )
        page.update()

    title = ft.Text(
        "AGENT SOARM100",
        size=40,
        weight=ft.FontWeight.BOLD,
        font_family="Courier",
        color=ft.colors.BLACK,
        text_align=ft.TextAlign.CENTER,
    )

    # Reemplazar el TextField con ListView
    log_area = ft.ListView(
        expand=True,
        spacing=10,
        auto_scroll=True,
        padding=10,
    )

    # Contenedor para el ListView con borde
    log_container = ft.Container(
        content=log_area,
        width=800,
        height=400,
        border=ft.border.all(2, ft.colors.BLACK),
        bgcolor=ft.colors.WHITE,
        padding=10,
    )

    mic_button = ft.IconButton(
        icon=ft.icons.MIC,
        icon_color=ft.colors.BLACK,
        icon_size=100,
        tooltip="Start recording",
        style=ft.ButtonStyle(
            shape=ft.CircleBorder(),
            side=ft.BorderSide(2, ft.colors.BLACK),
        ),
        on_click=toggle_recording,
    )

    nonhuman_label = ft.Text(
        "NONHUMAN",
        size=14,
        font_family="PressStart2P",
        color=ft.colors.BLACK,
        text_align=ft.TextAlign.CENTER,
    )

    page.add(
        ft.Row([
            ft.Container(expand=True),
            theme_toggle
        ]),
        ft.Column([
            ft.Container(content=title, alignment=ft.alignment.center),
            ft.Container(content=log_container, alignment=ft.alignment.center, expand=True),
            ft.Container(content=mic_button, alignment=ft.alignment.center, margin=ft.margin.only(bottom=20)),
            ft.Container(content=nonhuman_label, alignment=ft.alignment.center),
        ], alignment=ft.MainAxisAlignment.CENTER, expand=True, spacing=20)
    )

    # Example of adding a log message
    add_log("System initialized...")

ft.app(target=main)