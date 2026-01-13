#!/usr/bin/env python3
"""
Quick Test - Token Buyer Discovery
Ejecuta el discovery una vez para probarlo
"""

import asyncio
import sys
import io

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from token_buyer_discovery import TokenBuyerDiscovery


async def main():
    print("=" * 70)
    print("TOKEN BUYER DISCOVERY - TEST")
    print("=" * 70)
    print("\nAnalizando últimas 24h de compras de KOLs...")
    print("Esto puede tomar varios minutos...\n")

    discovery = TokenBuyerDiscovery()

    results = await discovery.analyze_recent_kol_buys(hours=24)

    print("\n" + "=" * 70)
    print("RESUMEN FINAL")
    print("=" * 70)
    print(f"\nTokens analizados: {results['analyzed']}")
    print(f"Traders descubiertos: {results['discovered']}")
    print(f"Promovidos a KOL: {results['promoted']}")

    if results['discovered'] > 0:
        print(f"\n[+] EXITO: Descubriste {results['discovered']} nuevos traders!")
    else:
        print(f"\n[i] No se descubrieron nuevos traders esta vez")


if __name__ == "__main__":
    asyncio.run(main())
