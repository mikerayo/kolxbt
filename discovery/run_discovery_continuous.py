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
from datetime import datetime

from token_buyer_discovery import TokenBuyerDiscovery


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

    parser.add_argument(
        '--hours-to-analyze',
        type=int,
        default=24,
        help='Hours of KOL buys to analyze (default: 24)'
    )

    args = parser.parse_args()

    # Fix Windows encoding
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    # Print banner
    print("=" * 70)
    print("CONTINUOUS TOKEN BUYER DISCOVERY")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Discovery interval: {args.interval} hours")
    print(f"Hours to analyze: {args.hours_to_analyze}")
    print(f"Mode: {'Run once' if args.once else 'Continuous'}")
    print("=" * 70)
    print()

    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Create discovery instance
    discovery = TokenBuyerDiscovery()

    async def run_discovery():
        """Run discovery once"""
        print(f"\n[*] Starting discovery run at {datetime.now().strftime('%H:%M:%S')}")
        print("-" * 70)

        results = await discovery.analyze_recent_kol_buys(hours=args.hours_to_analyze)

        print("\n" + "=" * 70)
        print("DISCOVERY SUMMARY")
        print("=" * 70)
        print(f"Tokens analyzed: {results['analyzed']}")
        print(f"New traders discovered: {results['discovered']}")
        print(f"Promoted to KOL: {results['promoted']}")

        if results['top_discovered']:
            print("\nTop Discovered Traders:")
            for i, trader in enumerate(results['top_discovered'][:5], 1):
                print(f"  {i}. {trader['wallet'][:8]}... - Score: {trader['score']:.0f}")

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
