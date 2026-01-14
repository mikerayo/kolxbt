#!/usr/bin/env python3
"""
Generate simple performance report from ClosedPositions
"""
import os
os.environ['DATABASE_URL'] = 'postgresql://kol_tracker_db_user:quNTItA4CvEhk9KmsK1irizJQQGWu99X@dpg-d5jcq924d50c73fqo9t0-a.virginia-postgres.render.com/kol_tracker_db'

from core.database import db, ClosedPosition, KOL
from datetime import datetime

def main():
    session = db.get_session()

    print("=" * 70)
    print("KOL TRACKER ML - PERFORMANCE REPORT")
    print("=" * 70)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Get all closed positions
    cps = session.query(ClosedPosition).all()

    if not cps:
        print("[ERROR] No closed positions found")
        return

    # Calculate metrics
    total_positions = len(cps)
    profitable = sum(1 for cp in cps if cp.pnl_multiple > 1.0)
    unprofitable = total_positions - profitable

    avg_pnl = sum(cp.pnl_multiple for cp in cps) / total_positions
    avg_hold_time = sum(cp.hold_time_seconds for cp in cps) / total_positions

    best_trade = max(cps, key=lambda cp: cp.pnl_multiple)
    worst_trade = min(cps, key=lambda cp: cp.pnl_multiple)

    # Calculate total return if following all trades
    total_invested = sum(cp.entry_price * cp.hold_time_seconds for cp in cps if cp.entry_price)  # Simplified
    total_return = sum(cp.pnl_multiple for cp in cps) / total_positions

    # Print metrics
    print("[SUMMARY METRICS]")
    print(f"  Total Closed Positions: {total_positions}")
    print(f"  Profitable Trades: {profitable} ({profitable/total_positions*100:.1f}%)")
    print(f"  Unprofitable Trades: {unprofitable} ({unprofitable/total_positions*100:.1f}%)")
    print()

    print("[PERFORMANCE METRICS]")
    print(f"  Average PnL Multiple: {avg_pnl:.2f}x")
    print(f"  Average Hold Time: {avg_hold_time/60:.1f} minutes")
    print(f"  Total Return: {total_return:.2f}x")
    print()

    print("[BEST & WORST TRADES]")
    print(f"  Best Trade: {best_trade.pnl_multiple:.2f}x")
    print(f"  Worst Trade: {worst_trade.pnl_multiple:.2f}x")
    print()

    # Top performers
    print("[TOP 5 TRADES]")
    sorted_cps = sorted(cps, key=lambda cp: cp.pnl_multiple, reverse=True)[:5]
    for i, cp in enumerate(sorted_cps, 1):
        print(f"  {i}. {cp.pnl_multiple:.2f}x | Hold: {cp.hold_time_seconds/60:.1f}min | {cp.entry_time.strftime('%Y-%m-%d %H:%M')}")

    print()
    print("[CONCLUSIONS]")
    if avg_pnl > 2.0:
        print("  [EXCELLENT] Average return > 2x")
    elif avg_pnl > 1.5:
        print("  [GOOD] Average return > 1.5x")
    else:
        print("  [NEEDS IMPROVEMENT] Average return < 1.5x")

    if profitable/total_positions > 0.6:
        print(f"  [EXCELLENT] Win rate > 60% ({profitable/total_positions*100:.1f}%)")
    elif profitable/total_positions > 0.5:
        print(f"  [GOOD] Win rate > 50% ({profitable/total_positions*100:.1f}%)")
    else:
        print(f"  [NEEDS IMPROVEMENT] Win rate < 50% ({profitable/total_positions*100:.1f}%)")

    print()
    print("[RECOMMENDATIONS]")
    print("  Based on the data:")
    if avg_pnl > 2.0 and profitable/total_positions > 0.6:
        print("  - Following top KOLs shows STRONG performance")
        print("  - Consider increasing position sizes")
        print("  - Continue monitoring these KOLs")
    else:
        print("  - More data needed for conclusive recommendations")
        print("  - Continue tracking for 1-2 weeks")
        print("  - Analyze which KOLs perform best")

    session.close()

if __name__ == "__main__":
    main()
