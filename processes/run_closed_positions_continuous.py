#!/usr/bin/env python3
"""
Continuous Closed Positions Creator
Periodically pairs buys with sells to create ClosedPositions

Runs every 10 minutes
"""

import asyncio
import time
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.database import db, Trade, ClosedPosition
from sqlalchemy import func


async def create_closed_positions():
    """
    Pair buys with sells to create ClosedPositions
    """
    session = db.get_session()

    try:
        # Get all trades
        all_trades = session.query(Trade).order_by(Trade.timestamp).all()

        if not all_trades:
            return 0

        # Get existing closed positions to avoid duplicates
        existing_cps = session.query(ClosedPosition).all()
        existing_keys = {(cp.kol_id, cp.token_address, cp.entry_time, cp.exit_time) for cp in existing_cps}

        # Group by (kol_id, token_address)
        trade_groups = defaultdict(list)
        for trade in all_trades:
            key = (trade.kol_id, trade.token_address)
            trade_groups[key].append(trade)

        # Process each group
        closed_positions_created = 0

        for (kol_id, token_address), trades in trade_groups.items():
            # Sort by timestamp
            trades.sort(key=lambda t: t.timestamp)

            # Pair buys with sells (FIFO)
            buy_queue = []  # List of buy trades waiting to be closed

            for trade in trades:
                if trade.operation == 'buy':
                    buy_queue.append(trade)
                elif trade.operation == 'sell' and buy_queue:
                    # Match this sell with earliest buy
                    buy_trade = buy_queue.pop(0)

                    # Check if this closed position already exists
                    cp_key = (kol_id, token_address, buy_trade.timestamp, trade.timestamp)
                    if cp_key in existing_keys:
                        continue

                    # Calculate PnL using prices
                    entry_price = buy_trade.price or 0
                    exit_price = trade.price or 0

                    # PnL calculation
                    entry_amount_sol = buy_trade.amount_sol
                    exit_amount_sol = trade.amount_sol
                    pnl_sol = exit_amount_sol - entry_amount_sol
                    pnl_multiple = exit_amount_sol / entry_amount_sol if entry_amount_sol > 0 else 1.0

                    # Calculate hold time
                    hold_time_seconds = (trade.timestamp - buy_trade.timestamp).total_seconds()

                    # Create ClosedPosition
                    closed = ClosedPosition(
                        kol_id=kol_id,
                        token_address=token_address,
                        entry_time=buy_trade.timestamp,
                        exit_time=trade.timestamp,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        pnl_sol=pnl_sol,
                        pnl_multiple=pnl_multiple,
                        hold_time_seconds=hold_time_seconds,
                        dex=buy_trade.dex
                    )

                    session.add(closed)
                    closed_positions_created += 1

        # Commit all changes
        session.commit()

        return closed_positions_created

    except Exception as e:
        session.rollback()
        print(f"[!] Error creating closed positions: {e}")
        import traceback
        traceback.print_exc()
        return 0
    finally:
        session.close()


async def closed_positions_loop():
    """
    Continuous loop to create closed positions
    Runs every 10 minutes
    """
    interval_seconds = 10 * 60  # 10 minutes

    print("[*] Closed Positions Creator started")
    print(f"[*] Creating closed positions every {interval_seconds//60} minutes")

    while True:
        try:
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Creating closed positions...")

            created = await create_closed_positions()

            if created > 0:
                print(f"[+] Created {created} new closed positions")
            else:
                print(f"[*] No new closed positions to create")

            # Get total count
            session = db.get_session()
            total = session.query(ClosedPosition).count()
            session.close()

            print(f"[*] Total closed positions in database: {total}")

            # Wait for next iteration
            print(f"[*] Waiting {interval_seconds//60} minutes until next run...")
            await asyncio.sleep(interval_seconds)

        except Exception as e:
            print(f"[!] Error in closed positions loop: {e}")
            import traceback
            traceback.print_exc()
            # Wait 1 minute before retrying
            await asyncio.sleep(60)


def main():
    """
    Entry point for the closed positions creator
    """
    try:
        asyncio.run(closed_positions_loop())
    except KeyboardInterrupt:
        print("\n[!] Closed Positions Creator stopped by user")


if __name__ == "__main__":
    main()
