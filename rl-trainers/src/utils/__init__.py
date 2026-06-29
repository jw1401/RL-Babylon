import sys
from .logger import Logger
from .config_loader import ConfigLoader
from .start import run_envs

# Prüfen, welches Betriebssystem läuft
if sys.platform == "darwin":
    # Wenn es ein Mac (Apple OS X / macOS) ist
    from .keyboard_mac import Keyboard
else:
    # Für Windows (und als Fallback für Linux/andere)
    from .keyboard import Keyboard