"""
KOL Tracker ML System - Master Launcher

Arranca todos los procesos del sistema en segundo plano con un solo comando.

Procesos:
- Tracker: Escanea trades cada 5 minutos
- ML Trainer: Reentrena modelos cada 1 hora
- Token Discovery: Descubre nuevos traders cada 1 hora
- Token Updater: Actualiza metadata cada 35 minutos
- Dashboard: Interfaz web unificada

Uso:
    python start_all.py
"""

import subprocess
import sys
import time
import os
import signal
import io
from datetime import datetime

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Lista de procesos
processes = []

def print_header():
    """Imprime header del sistema"""
    print("\n" + "=" * 70)
    print("💎 KOL TRACKER ML SYSTEM - MASTER LAUNCHER")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")

def print_summary():
    """Imprime resumen de procesos"""
    print("\n" + "=" * 70)
    print("📊 PROCESOS ACTIVOS")
    print("=" * 70)
    print(f"{'Proceso':<25} {'PID':<10} {'Intervalo':<15} {'Status'}")
    print("-" * 70)
    print(f"{'🔍 Tracker':<25} {'N/A':<10} {'5 minutos':<15} {'✅ Running'}")
    print(f"{'🧠 ML Trainer':<25} {'N/A':<10} {'1 hora':<15} {'✅ Running'}")
    print(f"{'🕵️ Token Discovery':<25} {'N/A':<10} {'1 hora':<15} {'✅ Running'}")
    print(f"{'🪙 Token Updater':<25} {'N/A':<10} {'35 minutos':<15} {'✅ Running'}")
    print(f"{'📊 Dashboard':<25} {'N/A':<10} {'Auto':<15} {'✅ Running'}")
    print("=" * 70)

def start_process(name, command, log_file):
    """
    Inicia un proceso en segundo plano

    Args:
        name: Nombre del proceso
        command: Comando a ejecutar
        log_file: Archivo de log

    Returns:
        subprocess.Popen object
    """
    print(f"[+] Iniciando {name}...")

    # Abrir archivo de log
    log_handle = open(log_file, 'a', encoding='utf-8')

    # Iniciar proceso
    process = subprocess.Popen(
        command,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True
    )

    processes.append({
        'name': name,
        'process': process,
        'log_file': log_file
    })

    print(f"    ✓ {name} iniciado (PID: {process.pid})")
    print(f"    Log: {log_file}")

    return process

def stop_all_processes(signum=None, frame=None):
    """
    Detiene todos los procesos activos

    Args:
        signum: Signal number (opcional)
        frame: Current stack frame (opcional)
    """
    print("\n" + "=" * 70)
    print("🛑 DETENIENDO TODOS LOS PROCESOS...")
    print("=" * 70)

    for proc_info in processes:
        try:
            proc = proc_info['process']
            name = proc_info['name']

            if proc.poll() is None:  # Process still running
                print(f"\n[+] Deteniendo {name} (PID: {proc.pid})...")

                if sys.platform == 'win32':
                    proc.terminate()
                else:
                    proc.send_signal(signal.SIGTERM)

                # Wait for process to terminate
                try:
                    proc.wait(timeout=5)
                    print(f"    ✓ {name} detenido correctamente")
                except subprocess.TimeoutExpired:
                    print(f"    ! {name} no respondió, forzando cierre...")
                    proc.kill()
                    proc.wait()
                    print(f"    ✓ {name} forzado a cerrar")

            # Close log file
            if 'log_file' in proc_info:
                log_file = proc_info['log_file']
                if not log_file.closed:
                    log_file.close()

        except Exception as e:
            print(f"    ✗ Error deteniendo {proc_info['name']}: {e}")

    print("\n" + "=" * 70)
    print("✓ TODOS LOS PROCESOS DETENIDOS")
    print("=" * 70)

    # Exit
    sys.exit(0)

def main():
    """Función principal"""

    print_header()

    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print("[*] Directorio de trabajo:")
    print(f"    {script_dir}\n")

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, stop_all_processes)
    if sys.platform != 'win32':
        signal.signal(signal.SIGTERM, stop_all_processes)

    # ============================================
    # INICIAR PROCESOS EN ORDEN
    # ============================================

    print("\n[*] Iniciando procesos...\n")

    # 1. Tracker (5 minutos)
    start_process(
        "Tracker",
        [sys.executable, "processes/run_tracker_continuous.py"],
        "tracker.log"
    )
    time.sleep(2)  # Wait between processes

    # 2. ML Trainer (1 hora)
    start_process(
        "ML Trainer",
        [sys.executable, "processes/run_continuous_trainer.py"],
        "trainer.log"
    )
    time.sleep(2)

    # 3. Token Discovery (1 hora)
    start_process(
        "Token Discovery",
        [sys.executable, "discovery/run_token_discovery_continuous.py"],
        "discovery.log"
    )
    time.sleep(2)

    # 4. Token Updater (35 minutos) - DexScreener + Bubblemaps
    start_process(
        "Token Updater",
        [sys.executable, "processes/run_token_updater_both_continuous.py"],
        "token_updater_both.log"
    )
    time.sleep(2)

    # 5. Dashboard
    print(f"[+] Iniciando Dashboard...")
    print(f"    Iniciando dashboard en http://localhost:8502")

    dashboard_log = open("dashboard.log", 'a', encoding='utf-8')
    dashboard_process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "dashboard/dashboard_unified.py", "--server.headless", "true"],
        stdout=dashboard_log,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True
    )

    processes.append({
        'name': 'Dashboard',
        'process': dashboard_process,
        'log_file': 'dashboard.log'
    })

    print(f"    ✓ Dashboard iniciado (PID: {dashboard_process.pid})")
    print(f"    Log: dashboard.log")

    time.sleep(3)

    # ============================================
    # RESUMEN
    # ============================================

    print_summary()

    print("\n📊 DASHBOARD DISPONIBLE:")
    print("    URL: http://localhost:8502")
    print("    Network: http://192.168.1.174:8502")

    print("\n⏰ INTERVALOS DE ACTUALIZACIÓN:")
    print("    Tracker:       5 minutos")
    print("    ML Trainer:    1 hora")
    print("    Token Discovery: 1 hora")
    print("    Token Updater: 35 minutos (DexScreener + Bubblemaps)")

    print("\n📝 LOGS:")
    print("    tracker.log            - Tracker activity")
    print("    trainer.log            - ML training logs")
    print("    discovery.log          - Token discovery logs")
    print("    token_updater_both.log - Token metadata updates (DexScreener + Bubblemaps)")
    print("    dashboard.log          - Streamlit dashboard")

    print("\n" + "=" * 70)
    print("✅ SISTEMA INICIADO CORRECTAMENTE")
    print("=" * 70)
    print("\n[*] Presiona Ctrl+C para detener todos los procesos\n")

    # ============================================
    # MONITOREO DE PROCESOS
    # ============================================

    print("[*] Monitorizando procesos (Ctrl+C para detener)...\n")

    try:
        while True:
            # Check if any process died
            for proc_info in processes:
                proc = proc_info['process']
                name = proc_info['name']

                if proc.poll() is not None:
                    print(f"\n[!] {name} (PID: {proc.pid}) ha dejado de ejecutarse!")
                    print(f"    Exit code: {proc.poll()}")
                    print(f"    Log: {proc_info['log_file']}")

            # Wait before next check
            time.sleep(30)

    except KeyboardInterrupt:
        # This will be caught by signal handler
        pass

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        stop_all_processes()
