"""
Hot KOLs Viewer - Muestra los KOLs más activos en tiempo real
"""

import sys
import io
from datetime import datetime

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from hot_kols_scorer import HotKOLsScorer


def main():
    print("\n" + "=" * 70)
    print(" " * 20 + "HOT KOLS RANKING")
    print("=" * 70)
    print(f"\nActualizado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Periodo: Ultimas 24 horas\n")

    scorer = HotKOLsScorer()
    hot_kols = scorer.get_hot_kols(top_n=15, hours=24)

    if not hot_kols:
        print("[!] No hay KOLs con actividad reciente")
        return

    print(f"{'#':<3} {'KOL':<20} {'Wallet':<20} {'Score':<7} {'Trades':<7} {'Vol (SOL)':<10}")
    print("-" * 80)

    for i, kol in enumerate(hot_kols, 1):
        name = kol['name'][:20] if kol['name'] else 'Unknown'
        wallet = kol['short_address']
        score = kol['score']
        trades = kol['breakdown']['recent_trades']
        vol = kol['breakdown']['recent_volume']

        # Color code by score
        if score >= 80:
            symbol = "[ELITE]"
        elif score >= 70:
            symbol = "[HOT]  "
        elif score >= 60:
            symbol = "[GOOD] "
        else:
            symbol = "[OK]   "

        print(f"{i:<3} {name:<20} {wallet:<20} {score:<7.1f} {trades:<7} {vol:<10.2f} {symbol}")

    print("\n" + "=" * 70)
    print("DETALLE DEL TOP KOL")
    print("=" * 70)

    top_kol = hot_kols[0]
    summary = scorer.get_kol_activity_summary(top_kol['kol_id'], hours=24)

    print(f"\nKOL: {top_kol['name']} ({top_kol['short_address']})")
    print(f"\nActividad (ultimas 24h):")
    print(f"  Total trades: {summary['total_trades']}")
    print(f"  Buys: {summary['buys']} | Sells: {summary['sells']}")
    print(f"  Tokens unicos: {summary['unique_tokens']}")
    print(f"  Volumen total: {summary['total_volume_sol']:.2f} SOL")
    print(f"  Tamano promedio: {summary['avg_trade_size_sol']:.2f} SOL")

    print(f"\nScore Breakdown:")
    b = top_kol['breakdown']
    print(f"  Trades (30%):     {b['trades_score']:.1f}/100")
    print(f"  Volume (40%):     {b['volume_score']:.1f}/100")
    print(f"  Win Rate (20%):   {b['win_rate_score']:.1f}/100")
    print(f"  DH Rate (10%):    {b['dh_score']:.1f}/100")
    print(f"  FINAL SCORE:      {b['hot_score']:.1f}/100")


if __name__ == "__main__":
    main()
