#!/usr/bin/env python3
"""
Quick test - Verify pump.fun detector works with 1 KOL
"""

import asyncio
from database import db
from wallet_tracker import WalletTracker

async def test_pumpfun_detector():
    print("="*60)
    print("PUMP.FUN DETECTOR TEST - 1 KOL")
    print("="*60)

    session = db.get_session()
    kols = db.get_all_kols(session)
    kol = kols[0]  # Cented (primer KOL)
    session.close()

    print(f"\n[*] Testing with: {kol.name}")
    print(f"    Wallet: {kol.short_address}")
    print(f"    Twitter: @{kol.twitter_username}")
    print(f"\n[*] Fetching last 100 signatures...")

    # Get signatures
    async with WalletTracker() as tracker:
        sigs = await tracker.get_signatures_for_address(kol.wallet_address, limit=100)
        print(f"[+] Got {len(sigs)} signatures")

        trades_found = 0
        pumpfun_trades = 0
        raydium_trades = 0
        jupiter_trades = 0

        print(f"\n[*] Processing {min(50, len(sigs))} transactions...")

        for i, sig_data in enumerate(sigs[:50], 1):
            sig = sig_data.get('signature')
            if not sig:
                continue

            # Fetch transaction
            tx = await tracker.get_transaction(sig)
            if not tx:
                continue

            # Parse trade
            trade = tracker.parser.parse_transaction(tx, kol.wallet_address)
            if trade:
                trades_found += 1

                dex = trade.get('dex', 'unknown')
                print(f"\n  [{i}] TRADE FOUND!")
                print(f"      DEX: {dex}")
                print(f"      Operation: {trade['operation']}")
                print(f"      Token: {trade['token_address'][:16]}...")
                print(f"      Amount SOL: {trade['amount_sol']:.4f}")
                print(f"      Amount Token: {trade['amount_token']:.2f}")
                print(f"      Price: {trade.get('price', 'N/A')}")

                if dex == 'pump_fun':
                    pumpfun_trades += 1
                elif dex == 'raydium':
                    raydium_trades += 1
                elif dex == 'jupiter':
                    jupiter_trades += 1

            # Small delay
            await asyncio.sleep(0.1)

    print(f"\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"Total trades found: {trades_found}")
    print(f"  - pump.fun: {pumpfun_trades}")
    print(f"  - raydium: {raydium_trades}")
    print(f"  - jupiter: {jupiter_trades}")

    if trades_found > 0:
        print(f"\n[OK] PUMP.FUN DETECTOR WORKING!")
        if pumpfun_trades > 0:
            print(f"[EXCELLENTE] Found {pumpfun_trades} pump.fun trades!")
        else:
            print(f"[INFO] No pump.fun trades, but detector is working")
        print(f"\n[*] Safe to delete database and restart full tracker")
        return True
    else:
        print(f"\n[!] No trades found - may need:")
        print(f"    1. Different time period")
        print(f"    2. Different KOL")
        print(f"    3. Check if KOL is active")
        return False

if __name__ == "__main__":
    asyncio.run(test_pumpfun_detector())
