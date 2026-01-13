"""
KOL Tracker ML System - Stop All Processes

Detiene todos los procesos del sistema de forma segura.

Uso:
    python stop_all.py
"""

import subprocess
import sys
import os

def kill_processes_by_name():
    """Mata procesos por nombre"""

    print("\n" + "=" * 70)
    print("🛑 DETENIENDO PROCESOS DEL KOL TRACKER")
    print("=" * 70 + "\n")

    processes_to_kill = [
        "python.*run_tracker_continuous",
        "python.*run_continuous_trainer",
        "python.*run_token_discovery_continuous",
        "python.*run_token_updater_continuous",
        "streamlit.*dashboard_unified",
        "python.*-m.*streamlit"
    ]

    killed_count = 0

    if sys.platform == 'win32':
        # Windows
        print("[*] Buscando procesos en Windows...")

        # Kill Python processes running our scripts
        for pattern in processes_to_kill:
            try:
                result = subprocess.run(
                    ['taskkill', '/F', '/IM', 'python.exe', '/FI', f'WINDOWTITLE eq {pattern}'],
                    capture_output=True,
                    text=True
                )
            except:
                pass

        # Kill streamlit
        try:
            result = subprocess.run(
                ['taskkill', '/F', '/IM', 'streamlit.exe'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0 or 'not found' not in result.stderr.lower():
                print(f"    ✓ streamlit.exe terminado")
                killed_count += 1
        except Exception as e:
            print(f"    ✗ Error terminando streamlit: {e}")

        # Kill all python processes running our scripts (more aggressive)
        try:
            # Get list of Python processes
            result = subprocess.run(
                ['tasklist', '/FI', 'IMAGENAME eq python.exe'],
                capture_output=True,
                text=True
            )

            if 'python.exe' in result.stdout:
                print("\n[+] Procesos Python encontrados, terminando...")
                result = subprocess.run(
                    ['taskkill', '/F', '/IM', 'python.exe'],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    print(f"    ✓ Procesos Python terminados")
                    killed_count += 1
        except Exception as e:
            print(f"    ✗ Error: {e}")

    else:
        # Linux/Mac
        print("[*] Buscando procesos en Linux/Mac...")

        for pattern in processes_to_kill:
            try:
                # Find processes
                result = subprocess.run(
                    ['pgrep', '-f', pattern],
                    capture_output=True,
                    text=True
                )

                if result.returncode == 0 and result.stdout.strip():
                    pids = result.stdout.strip().split('\n')

                    for pid in pids:
                        try:
                            subprocess.run(['kill', '-TERM', pid], check=True)
                            print(f"    ✓ Proceso {pid} terminado")
                            killed_count += 1
                        except subprocess.CalledProcessError:
                            # Force kill if SIGTERM fails
                            try:
                                subprocess.run(['kill', '-9', pid], check=True)
                                print(f"    ✓ Proceso {pid} forzado")
                                killed_count += 1
                            except:
                                pass

            except Exception as e:
                pass

    print("\n" + "=" * 70)
    print(f"✓ PROCESOS DETENIDOS: {killed_count}")
    print("=" * 70)

    if killed_count == 0:
        print("\n[i] No se encontraron procesos activos del sistema")
    else:
        print(f"\n[i] {killed_count} proceso(s) terminado(s)")

    print("\n[i] Puedes verificar que no quedan procesos con:")
    if sys.platform == 'win32':
        print("    tasklist | findstr python")
        print("    tasklist | findstr streamlit")
    else:
        print("    ps aux | grep python")
        print("    ps aux | grep streamlit")

    print()

if __name__ == "__main__":
    kill_processes_by_name()
