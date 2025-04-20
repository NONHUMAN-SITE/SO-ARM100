from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from datetime import datetime

class Logger:
    def __init__(self):
        self.console = Console()

    def log(self, message, level='info'):
        # Define los colores para cada nivel
        colors = {
            'success': 'green',
            'error': 'red',
            'warning': 'yellow',
            'debug': 'cyan',
            'info': 'white'
        }
        color = colors.get(level, 'white')
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # Crea el texto con el color correspondiente
        text = Text(message, style=color)
        # Crea un panel con el texto y el título como la marca de tiempo
        panel = Panel(text, title=timestamp, border_style=color)
        self.console.print(panel)

# Ejemplo de uso
logger = Logger()
logger.log("Este es un mensaje de éxito", level="success")
logger.log("Algo salió mal", level="error")
logger.log("Esto es una advertencia", level="warning")
logger.log("Mensaje de depuración", level="debug")
logger.log("Mensaje informativo", level="info")
