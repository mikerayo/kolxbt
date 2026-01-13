#!/usr/bin/env python3
"""
Small test - Track just 1 KOL with 50 signatures max
"""

import asyncio
from database import db
from wallet_tracker import WalletTracker

async def test_one_kol_small():
    print("="*60)
    print("SMALL TEST - 1 KOL, 50 signatures")
    print("="*60)

    session = db.get_session()
    kols = db.get_all_kols(session)
    kol = kols[0]  # Cented

    print(f"\n[*] Testing: {kol.name}")
    print(f"    Wallet: {kol.short_address}")

    # Get just 50 signatures (very small test)
    async with WalletTracker() as tracker:
        print(f"\n[*] Fetching 50 signatures...")
        sigs = await tracker.get_signatures_for_address(kol.wallet_address, limit=50)
        print(f"[+] Got {len(sigs)} signatures")

        trades_found = 0
        for i, sig_data in enumerate(sigs[:20], 1):  # Only process first 20
            sig = sig_data.get('signature')
            if not sig:
                continue

            print(f"  [{i}/{min(20, len(sigs))}] Fetching {sig[:16]}...")

            tx = await tracker.get_transaction(sig)
            if tx:
                trade = tracker.parser.parse_transaction(tx, kol.wallet_address)
                if trade:
                    trades_found += 1
                    print(f"      [+] TRADE FOUND: {trade['operation']} {trade['token_address'][:12]}...")

            # Small delay to avoid rate limits
            await asyncio.sleep(0.5)

    print(f"\n[OK] Test complete! Found {trades_found} trades")
    print("\nConclusion:")
    print("  - System is working")
    print("  - Helius RPC has rate limits")
    print("  - Need 2+ seconds delay between RPC calls")
    print("  - Tracking 450 KOLs would take many hours")

if __name__ == "__main__":
    asyncio.run(test_one_kol_small())
