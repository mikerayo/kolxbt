#!/usr/bin/env python3
"""
Create ClosedPositions from existing Trades
Empareja buys con sells para crear posiciones cerradas
"""
import os
import sys
from pathlib import Path
from collections import defaultdict

# Set DATABASE_URL BEFORE importing
os.environ['DATABASE_URL'] = 'postgresql://kol_tracker_db_user:quNTItA4CvEhk9KmsK1irizJQQGWu99X@dpg-d5jcq924d50c73fqo9t0-a.virginia-postgres.render.com/kol_tracker_db'

from core.database import db, Trade, ClosedPosition, KOL
from sqlalchemy import and_

def main():
    session = db.get_session()

    print("=" * 70)
    print("CREATE CLOSED POSITIONS FROM TRADES")
    print("=" * 70)

    # Get all trades
    all_trades = session.query(Trade).order_by(Trade.timestamp).all()
    print(f"\nTotal trades: {len(all_trades)}")

    # Group by (kol_id, token_address)
    trade_groups = defaultdict(list)
    for trade in all_trades:
        key = (trade.kol_id, trade.token_address)
        trade_groups[key].append(trade)

    print(f"Unique (KOL, token) pairs: {len(trade_groups)}")

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

                # Calculate PnL using prices
                # Entry: bought tokens with buy_trade.amount_sol
                # Exit: sold tokens for trade.amount_sol
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

                print(f"[+] Created: KOL {kol_id} | {token_address[:8]}... | {pnl_multiple:.2f}x | {hold_time_seconds/3600:.1f}h")

        # Any remaining buys are still open
        if buy_queue:
            print(f"[*] KOL {kol_id} | {token_address[:8]}... | {len(buy_queue)} buys still open")

    # Commit all changes
    session.commit()

    print(f"\n" + "=" * 70)
    print(f"[DONE] Created {closed_positions_created} closed positions")
    print("=" * 70)

    # Summary stats
    closed_positions = session.query(ClosedPosition).all()

    if closed_positions:
        avg_pnl = sum(cp.pnl_multiple for cp in closed_positions) / len(closed_positions)
        profitable = sum(1 for cp in closed_positions if cp.pnl_multiple > 1.0)
        win_rate = profitable / len(closed_positions) * 100

        print(f"\n[STATS]")
        print(f"  Total positions: {len(closed_positions)}")
        print(f"  Avg PnL multiple: {avg_pnl:.2f}x")
        print(f"  Win rate: {win_rate:.1f}%")
        print(f"  Profitable trades: {profitable}")
        print(f"  Unprofitable trades: {len(closed_positions) - profitable}")

    session.close()

if __name__ == "__main__":
    main()
