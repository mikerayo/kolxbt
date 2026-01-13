"""
KOL Tracker ML - Punto de entrada principal

Usage:
    python main.py              # Arranca todo el sistema
    python main.py --help       # Muestra ayuda
"""
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from launchers.start_all import main as start_all_main

if __name__ == "__main__":
    start_all_main()
