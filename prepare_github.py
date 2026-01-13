"""
Script para preparar el proyecto para GitHub y Render
"""
import os
import subprocess
from pathlib import Path

def run_command(cmd, description):
    """Ejecuta un comando y muestra el resultado"""
    print(f"\n{'='*70}")
    print(f"[{description}]")
    print(f"{'='*70}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✓ {description} completado")
        if result.stdout:
            print(result.stdout)
    else:
        print(f"✗ Error en {description}")
        print(result.stderr)
    return result.returncode == 0

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║     🚀 KOL TRACKER ML - PREPARAR PARA GITHUB & RENDER              ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    # Verificar que git esté instalado
    if not run_command("git --version", "Verificando Git"):
        print("\n❌ Git no está instalado. Instala Git primero:")
        print("   https://git-scm.com/downloads")
        return

    # Verificar estado de git
    if not os.path.exists(".git"):
        print("\n📝 Inicializando repositorio Git...")
        if not run_command("git init", "Git init"):
            return
    else:
        print("\n✓ Git ya inicializado")

    # Añadir archivos
    print("\n📦 Añadiendo archivos al staging area...")
    run_command("git add .", "Git add")

    # Verificar qué se va a commit
    print("\n📋 Archivos preparados para commit:")
    run_command("git status", "Git status")

    # Pedir mensaje de commit
    commit_message = "feat: Initial commit - KOL Tracker ML with Bubblemaps integration\n\n- Complete project reorganization\n- Bubblemaps API integration\n- DexScreener API integration\n- Render deployment configuration\n- All processes ready for production"

    print(f"\n📝 Mensaje de commit:")
    print(f"   {commit_message[:50]}...")

    # Hacer commit
    if not run_command(f'git commit -m "{commit_message}"', "Git commit"):
        print("\n⚠️  No hay cambios para commit o ya existen commits")
        # Ver si ya hay commits previos
        result = subprocess.run("git log --oneline -n 1", shell=True, capture_output=True, text=True)
        if result.returncode == 0 and result.stdout:
            print("\n✓ Ya existe commit previo:")
            print(f"   {result.stdout.strip()}")

    # Preguntar si quiere configurar remote
    print("\n" + "="*70)
    print("📝 CONFIGURACIÓN GITHUB REMOTE")
    print("="*70)

    # Verificar si ya existe remote
    result = subprocess.run("git remote -v", shell=True, capture_output=True, text=True)
    if result.returncode == 0 and "origin" in result.stdout:
        print("\n✓ Remote 'origin' ya configurado:")
        print(f"   {result.stdout.strip()}")
    else:
        print("\n📌 Para configurar el repositorio remoto, sigue estos pasos:")
        print()
        print("   1. Crea un repositorio en GitHub:")
        print("      • Ve a https://github.com/new")
        print("      • Name: kol-tracker-ml")
        print("      • Private: ✓")
        print("      • NO marques 'Initialize with README'")
        print()
        print("   2. Ejecuta el siguiente comando:")
        print()
        print("      git remote add origin https://github.com/TU_USUARIO/kol-tracker-ml.git")
        print()
        print("   3. Sube el código:")
        print()
        print("      git branch -M main")
        print("      git push -u origin main")
        print()

    # Verificar archivos importantes
    print("\n" + "="*70)
    print("📋 ARCHIVOS CLAVE VERIFICADOS")
    print("="*70)

    archivos_clave = [
        ("render.yaml", "Configuración Render"),
        ("requirements.txt", "Dependencias Python"),
        (".gitignore", "Archivos ignorados"),
        ("README_RENDER.md", "Documentación deployment"),
        ("main.py", "Punto de entrada"),
        ("launchers/start_all.py", "Launcher principal"),
    ]

    for archivo, descripcion in archivos_clave:
        if os.path.exists(archivo):
            print(f"  ✓ {archivo:20s} - {descripcion}")
        else:
            print(f"  ✗ {archivo:20s} - FALTA")

    # Estructura de carpetas
    print("\n" + "="*70)
    print("📁 ESTRUCTURA DEL PROYECTO")
    print("="*70)

    carpetas = ["core", "apis", "processes", "discovery", "updaters", "dashboard", "launchers"]
    for carpeta in carpetas:
        if os.path.isdir(carpeta):
            num_archivos = len(list(Path(carpeta).glob("*.py")))
            print(f"  ✓ {carpeta:15s} - {num_archivos} archivos Python")

    print("\n" + "="*70)
    print("✅ PREPARACIÓN COMPLETADA")
    print("="*70)
    print()
    print("📝 PRÓXIMOS PASOS:")
    print()
    print("   1. Crea el repositorio en GitHub")
    print("   2. Configura el remote origin")
    print("   3. Haz push a GitHub:")
    print("      git push -u origin main")
    print()
    print("   4. Ve a Render.com y conecta el repo")
    print("   5. Deploy automático con render.yaml")
    print()
    print("📚 Consulta README_RENDER.md para instrucciones detalladas")
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Proceso interrumpido por el usuario")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
