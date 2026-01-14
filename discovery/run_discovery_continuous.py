#!/usr/bin/env python3
"""
Continuous Token Buyer Discovery - Corre automáticamente descubriendo nuevos traders

Ejecuta el descubrimiento de traders cada X horas:
- Analiza compras recientes de KOLs
- Busca wallets que compraron los mismos tokens
- Analiza su performance
- Guarda los mejores en discovered_traders
- Auto-promueve a KOL si score >= 80
"""

import asyncio
import argparse
import signal
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from discovery.token_centric_discovery import TokenHolderDiscovery


def signal_handler(sig, frame):
    """Handle interrupt signals"""
    print("\n[!] Interrupt received, stopping discovery...")
    sys.exit(0)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Continuous Token Buyer Discovery')

    parser.add_argument(
        '--interval',
        type=int,
        default=12,
        help='Discovery interval in hours (default: 12)'
    )

    parser.add_argument(
        '--once',
        action='store_true',
        help='Run discovery once and exit'
    )

    args = parser.parse_args()

    # Fix Windows encoding
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    # Print banner
    print("=" * 70)
    print("CONTINUOUS TOKEN HOLDER DISCOVERY")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Discovery interval: {args.interval} hours")
    print(f"Mode: {'Run once' if args.once else 'Continuous'}")
    print("=" * 70)
    print()

    # Setup signal handlers (only works in main thread)
    try:
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    except ValueError:
        # Running in a thread (orchestrator mode), signals handled by parent
        pass

    # Create discovery instance
    discovery = TokenHolderDiscovery()

    async def run_discovery():
        """Run discovery once"""
        print(f"\n[*] Starting discovery run at {datetime.now().strftime('%H:%M:%S')}")
        print("-" * 70)

        results = await discovery.discover_from_performing_tokens(min_kols=2, min_multiple=2.0)

        print("\n" + "=" * 70)
        print("DISCOVERY SUMMARY")
        print("=" * 70)
        print(f"Tokens analyzed: {results['tokens_analyzed']}")
        print(f"Performing tokens: {results['performing_tokens']}")
        print(f"New wallets discovered: {results['wallets_discovered']}")

        if results['wallets']:
            print("\nTop Discovered Traders:")
            for i, wallet in enumerate(results['wallets'][:5], 1):
                print(f"  {i}. {wallet['wallet'][:8]}... - {wallet['balance_tokens']:,.0f} tokens")

    if args.once:
        # Run once and exit
        asyncio.run(run_discovery())
        print("\n[+] Discovery complete - exiting")
        sys.exit(0)
    else:
        # Run continuous
        print("[*] Starting continuous discovery loop...")
        print("[*] Press Ctrl+C to stop\n")

        import time
        loop_count = 0

        while True:
            loop_count += 1

            print(f"\n{'='*70}")
            print(f"DISCOVERY LOOP #{loop_count}")
            print(f"{'='*70}")

            # Run discovery
            asyncio.run(run_discovery())

            # Wait before next run
            wait_seconds = args.interval * 3600
            print(f"\n[WAIT] Waiting {args.interval} hours until next discovery...")
            print(f"   Next run at: {(datetime.now()).strftime('%H:%M:%S')}")
            print(f"   Press Ctrl+C to stop\n")

            time.sleep(wait_seconds)


if __name__ == "__main__":
    main()
