"""
Continuous Token-Centric Discovery Runner

Descubre nuevos wallets traders analizando holders de tokens exitosos.
Corre continuamente en segundo plano.
"""

import asyncio
import time
import sys
import io
from datetime import datetime
from token_centric_discovery import TokenHolderDiscovery


# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


async def run_discovery_continuous(interval_hours=1):
    """
    Ejecuta el discovery de forma continua

    Args:
        interval_hours: Horas entre ejecuciones del discovery
    """
    discovery = TokenHolderDiscovery()

    print("=" * 70)
    print("CONTINUOUS TOKEN-CENTRIC DISCOVERY")
    print("=" * 70)
    print(f"\nIniciando discovery continuo...")
    print(f"Intervalo: {interval_hours} horas")
    print(f"Presiona Ctrl+C para detener\n")

    while True:
        try:
            print("\n" + "=" * 70)
            print(f"Discovery Run - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 70)

            # Run discovery
            # min_kols=5: Token debe ser comprado por al menos 5 KOLs
            # min_multiple=1.2: Token debe tener performance promedio >= 1.2x
            results = await discovery.discover_from_performing_tokens(
                min_kols=5,
                min_multiple=1.2
            )

            print("\n[✓] Discovery completado")
            print(f"    Tokens analizados: {results['tokens_analyzed']}")
            print(f"    Tokens exitosos: {results['performing_tokens']}")
            print(f"    Wallets descubiertas: {results['wallets_discovered']}")

            # Wait for next run
            print(f"\n[i] Próximo discovery en {interval_hours} horas...")
            print(f"   ({datetime.fromtimestamp(time.time() + interval_hours * 3600).strftime('%Y-%m-%d %H:%M:%S')})")

            await asyncio.sleep(interval_hours * 3600)

        except KeyboardInterrupt:
            print("\n\n[!] Deteniendo discovery...")
            break
        except Exception as e:
            print(f"\n[!] Error en discovery: {e}")
            import traceback
            traceback.print_exc()

            # Wait 5 minutes before retry
            print("[i] Reintentando en 5 minutos...")
            await asyncio.sleep(300)


def main():
    """Start continuous discovery"""
    # Run every 1 hour (can be adjusted)
    asyncio.run(run_discovery_continuous(interval_hours=1))


if __name__ == "__main__":
    main()
