#!/usr/bin/env python3
"""
Continuous KOL Tracker - Runs forever, scanning for new activity

Features:
- Scans all KOLs repeatedly
- Finds new trades since last scan
- Saves progress automatically
- Runs continuously until stopped
"""

import asyncio
import json
import time
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.database import db, Trade, KOL
from core.wallet_tracker import WalletTracker

# Progress file
PROGRESS_FILE = Path(__file__).parent / "data" / "tracking_progress.json"

# Configuration
SCAN_INTERVAL_MINUTES = 5  # Wait 5 minutes between full scans
DAYS_TO_SCAN = 1  # Only scan last 1 day for updates (faster)
MAX_SIGS = 500  # Max signatures per wallet


def load_progress():
    """Load tracking progress"""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {"completed": [], "last_update": None, "last_full_scan": None}


def save_progress(progress):
    """Save tracking progress"""
    progress["last_update"] = datetime.utcnow().isoformat()
    PROGRESS_FILE.parent.mkdir(exist_ok=True)
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


async def track_single_wallet(kol, tracker, session, days=1, max_sigs=500):
    """
    Track a single wallet

    Args:
        kol: KOL object
        tracker: WalletTracker instance
        session: Database session
        days: Days of history to scan
        max_sigs: Maximum signatures

    Returns:
        Number of NEW trades found
    """
    clean_name = kol.name.encode('ascii', 'ignore').decode('ascii') if kol.name else 'Unknown'

    try:
        # Get signatures
        sigs = await tracker.get_signatures_for_address(
            kol.wallet_address,
            limit=max_sigs
        )

        if not sigs:
            return 0

        # Process only new trades
        new_trades = 0
        batch_size = 50

        for i in range(0, min(len(sigs), max_sigs), batch_size):
            batch = sigs[i:i+batch_size]

            for sig_data in batch:
                sig = sig_data.get('signature')
                if not sig:
                    continue

                # Check if trade already exists
                existing = session.query(Trade).filter(
                    Trade.tx_signature == sig
                ).first()

                if existing:
                    continue

                # Fetch transaction
                tx = await tracker.get_transaction(sig)
                if not tx:
                    continue

                # Parse trade
                trade = tracker.parser.parse_transaction(tx, kol.wallet_address)
                if trade:
                    try:
                        db_trade = Trade(
                            kol_id=kol.id,
                            token_address=trade['token_address'],
                            operation=trade['operation'],
                            amount_sol=trade['amount_sol'],
                            amount_token=trade['amount_token'],
                            price=trade.get('price'),
                            dex=trade.get('dex'),
                            timestamp=trade['timestamp'],
                            tx_signature=trade['tx_signature']
                        )
                        session.add(db_trade)
                        session.commit()
                        new_trades += 1
                    except Exception as e:
                        session.rollback()
                        print(f"    [!] Error: {e}")

        return new_trades

    except Exception as e:
        print(f"[!] Error tracking {kol.name}: {e}")
        return 0


async def full_scan(session, tracker):
    """
    Perform a full scan of all KOLs (including promoted discovered traders)

    Returns:
        Total new trades found
    """
    print("\n" + "="*60)
    print(f"[SCAN] Starting full scan at {datetime.now().strftime('%H:%M:%S')}")
    print("="*60)

    # Get all KOLs
    all_kols = db.get_all_kols(session)

    # Check if any discovered traders have been promoted
    from database import DiscoveredTrader
    promoted_count = session.query(DiscoveredTrader).filter(
        DiscoveredTrader.promoted_to_kol == True
    ).count()

    tracking_count = session.query(DiscoveredTrader).filter(
        DiscoveredTrader.is_tracking == True,
        DiscoveredTrader.promoted_to_kol == False
    ).count()

    print(f"\n[*] Tracking {len(all_kols)} KOLs")
    if promoted_count > 0:
        print(f"    (+{promoted_count} promoted from discovery)")
    if tracking_count > 0:
        print(f"    ({tracking_count} discovered traders being monitored)")

    total_new_trades = 0
    wallets_with_new_trades = []

    async with tracker:
        for i, kol in enumerate(all_kols, 1):
            clean_name = kol.name.encode('ascii', 'ignore').decode('ascii') if kol.name else 'Unknown'

            print(f"\n[{i}/{len(all_kols)}] Scanning {clean_name}...")

            new_trades = await track_single_wallet(
                kol, tracker, session,
                days=DAYS_TO_SCAN,
                max_sigs=MAX_SIGS
            )

            if new_trades > 0:
                print(f"    [+] Found {new_trades} NEW trades!")
                total_new_trades += new_trades
                wallets_with_new_trades.append(kol.name)
            else:
                print(f"    [i] No new trades")

    print("\n" + "="*60)
    print(f"[COMPLETE] Scan finished - Found {total_new_trades} new trades")
    print("="*60)

    if wallets_with_new_trades:
        print(f"\n[INFO] KOLs with new activity: {', '.join(wallets_with_new_trades[:10])}")
        if len(wallets_with_new_trades) > 10:
            print(f"   ... and {len(wallets_with_new_trades) - 10} more")
    else:
        print("\n[INFO] No new activity found")

    return total_new_trades


async def continuous_tracking_loop():
    """
    Main continuous tracking loop
    """
    print("=" * 60)
    print("CONTINUOUS KOL TRACKER - RUNNING FOREVER")
    print("=" * 60)
    print("- Scans all KOLs repeatedly")
    print("- Finds new trades since last scan")
    print("- Waits between scans to avoid rate limits")
    print("- Press Ctrl+C to stop")
    print("=" * 60)
    print()

    session = db.get_session()
    scan_count = 0

    try:
        while True:
            scan_count += 1

            # Perform full scan
            new_trades = await full_scan(session, WalletTracker())

            # Update progress
            progress = load_progress()
            progress['last_full_scan'] = datetime.utcnow().isoformat()
            progress['scan_count'] = scan_count
            save_progress(progress)

            # Show stats
            print(f"\n[STATS]")
            print(f"   Scan #{scan_count}")
            print(f"   New trades this scan: {new_trades}")
            print(f"   Total KOLs: {len(db.get_all_kols(session))}")

            # Get database stats
            from sqlalchemy import func
            total_trades = session.query(func.count(Trade.id)).scalar()
            print(f"   Total trades in DB: {total_trades:,}")

            # Wait before next scan
            print(f"\n[WAIT] Waiting {SCAN_INTERVAL_MINUTES} minutes until next scan...")
            print(f"   Next scan at: {(datetime.now() + timedelta(minutes=SCAN_INTERVAL_MINUTES)).strftime('%H:%M:%S')}")
            print(f"   Press Ctrl+C to stop\n")

            await asyncio.sleep(SCAN_INTERVAL_MINUTES * 60)

    except KeyboardInterrupt:
        print("\n\n[!] Stopped by user")
        print("[+] Progress was saved!")
    finally:
        session.close()


async def main():
    """Main entry point"""
    try:
        await continuous_tracking_loop()
    except Exception as e:
        print(f"\n[!] Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n[!] Tracking stopped")
