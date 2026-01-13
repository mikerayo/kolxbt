import sys
"""
Hot KOLs Scorer - Identifica KOLs más activos y "calientes" en últimas 24h

Score compuesto por:
- 40%: Volumen de trades (últimas 24h)
- 30%: Número de trades (últimas 24h)
- 20%: Win rate reciente (últimas 48h)
- 10%: Diamond Hands rate (histórico)
"""

from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from sqlalchemy import func, and_
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.database import db, KOL, Trade, ClosedPosition


class HotKOLsScorer:
    """Calcula scores de KOLs basados en actividad reciente"""

    def __init__(self):
        self.session = db.get_session()

    def calculate_hot_score(self, kol_id: int, hours: int = 24) -> Tuple[float, Dict]:
        """
        Calcula Hot Score para un KOL

        Args:
            kol_id: ID del KOL
            hours: Horas hacia atrás para analizar (default: 24)

        Returns:
            Tuple[hot_score (0-100), breakdown_dict]
        """
        # Time window
        since = datetime.utcnow() - timedelta(hours=hours)

        # 1. Recent trades count (30% weight)
        recent_trades = self.session.query(func.count(Trade.id)).filter(
            and_(
                Trade.kol_id == kol_id,
                Trade.timestamp >= since
            )
        ).scalar() or 0

        trades_score = min(recent_trades / 20, 1.0) * 100  # Max score at 20 trades

        # 2. Recent volume (40% weight)
        recent_volume = self.session.query(func.sum(Trade.amount_sol)).filter(
            and_(
                Trade.kol_id == kol_id,
                Trade.timestamp >= since
            )
        ).scalar() or 0

        volume_score = min(recent_volume / 100, 1.0) * 100  # Max score at 100 SOL

        # 3. Recent win rate (20% weight) - look at closed positions from last 48h
        since_48h = datetime.utcnow() - timedelta(hours=48)
        recent_closed = self.session.query(ClosedPosition).filter(
            and_(
                ClosedPosition.kol_id == kol_id,
                ClosedPosition.exit_time >= since_48h
            )
        ).all()

        if recent_closed:
            wins = sum(1 for p in recent_closed if p.pnl_multiple > 1.0)
            win_rate = wins / len(recent_closed)
            win_rate_score = win_rate * 100
        else:
            # Use historical win rate if no recent closed positions
            historical_closed = self.session.query(ClosedPosition).filter(
                ClosedPosition.kol_id == kol_id
            ).all()

            if historical_closed:
                wins = sum(1 for p in historical_closed if p.pnl_multiple > 1.0)
                win_rate = wins / len(historical_closed)
                win_rate_score = win_rate * 100 * 0.5  # Discount historical
            else:
                win_rate_score = 0

        # 4. Diamond Hands rate (10% weight)
        all_closed = self.session.query(ClosedPosition).filter(
            ClosedPosition.kol_id == kol_id
        ).all()

        if all_closed:
            dh_count = sum(1 for p in all_closed if p.is_diamond_hand)
            dh_rate = dh_count / len(all_closed)
            dh_score = dh_rate * 100
        else:
            dh_score = 0

        # Calculate weighted score
        hot_score = (
            trades_score * 0.30 +
            volume_score * 0.40 +
            win_rate_score * 0.20 +
            dh_score * 0.10
        )

        breakdown = {
            'hot_score': round(hot_score, 2),
            'recent_trades': recent_trades,
            'trades_score': round(trades_score, 2),
            'recent_volume': round(recent_volume, 2),
            'volume_score': round(volume_score, 2),
            'win_rate_score': round(win_rate_score, 2),
            'dh_score': round(dh_score, 2),
            'period_hours': hours
        }

        return hot_score, breakdown

    def get_hot_kols(self, top_n: int = 10, hours: int = 24) -> List[Dict]:
        """
        Obtiene top N KOLs más calientes

        Args:
            top_n: Número de KOLs a retornar
            hours: Horas hacia atrás para analizar

        Returns:
            List of dicts with KOL info and scores
        """
        # Get all KOLs
        all_kols = self.session.query(KOL).all()

        scored_kols = []
        for kol in all_kols:
            score, breakdown = self.calculate_hot_score(kol.id, hours)

            # Only include KOLs with some activity
            if breakdown['recent_trades'] > 0 or breakdown['recent_volume'] > 0:
                scored_kols.append({
                    'kol_id': kol.id,
                    'name': kol.name or 'Unknown',
                    'wallet_address': kol.wallet_address,
                    'short_address': kol.short_address,
                    'score': score,
                    'breakdown': breakdown
                })

        # Sort by score descending
        scored_kols.sort(key=lambda x: x['score'], reverse=True)

        return scored_kols[:top_n]

    def get_kol_activity_summary(self, kol_id: int, hours: int = 24) -> Dict:
        """
        Obtiene resumen de actividad de un KOL

        Args:
            kol_id: ID del KOL
            hours: Horas hacia atrás

        Returns:
            Dict con resumen de actividad
        """
        since = datetime.utcnow() - timedelta(hours=hours)

        # Recent trades
        recent_trades = self.session.query(Trade).filter(
            and_(
                Trade.kol_id == kol_id,
                Trade.timestamp >= since
            )
        ).order_by(Trade.timestamp.desc()).all()

        # Group by operation
        buys = [t for t in recent_trades if t.operation == 'buy']
        sells = [t for t in recent_trades if t.operation == 'sell']

        # Unique tokens traded
        unique_tokens = set(t.token_address for t in recent_trades)

        # Total volume
        total_volume = sum(t.amount_sol for t in recent_trades)

        return {
            'period_hours': hours,
            'total_trades': len(recent_trades),
            'buys': len(buys),
            'sells': len(sells),
            'unique_tokens': len(unique_tokens),
            'total_volume_sol': round(total_volume, 2),
            'avg_trade_size_sol': round(total_volume / len(recent_trades), 2) if recent_trades else 0,
            'tokens_traded': list(unique_tokens)[:10]  # First 10
        }


def main():
    """Test the Hot KOLs scorer"""
    import sys
    import io

    # Fix Windows encoding
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("=" * 70)
    print("HOT KOLS RANKING - Últimas 24h")
    print("=" * 70)

    scorer = HotKOLsScorer()

    # Get top 10 hot KOLs
    hot_kols = scorer.get_hot_kols(top_n=10, hours=24)

    print(f"\nTop {len(hot_kols)} KOLs mas calientes:\n")

    for i, kol_data in enumerate(hot_kols, 1):
        print(f"{i}. {kol_data['name']} ({kol_data['short_address']})")
        print(f"   Hot Score: {kol_data['score']:.1f}/100")
        print(f"   Volumen: {kol_data['breakdown']['recent_volume']:.2f} SOL")
        print(f"   Trades: {kol_data['breakdown']['recent_trades']}")
        print()

    # Show detailed breakdown for top KOL
    if hot_kols:
        top_kol = hot_kols[0]
        print("=" * 70)
        print(f"TOP KOL DETALLE: {top_kol['name']}")
        print("=" * 70)

        summary = scorer.get_kol_activity_summary(top_kol['kol_id'], hours=24)

        print(f"\nActividad (ultimas {summary['period_hours']}h):")
        print(f"  Trades totales: {summary['total_trades']}")
        print(f"  Buys: {summary['buys']} | Sells: {summary['sells']}")
        print(f"  Tokens unicos: {summary['unique_tokens']}")
        print(f"  Volumen total: {summary['total_volume_sol']} SOL")
        print(f"  Tamano promedio trade: {summary['avg_trade_size_sol']} SOL")

        print(f"\nBreakdown del Score:")
        b = top_kol['breakdown']
        print(f"  Trades score: {b['trades_score']:.1f}/100 (30%)")
        print(f"  Volume score: {b['volume_score']:.1f}/100 (40%)")
        print(f"  Win rate score: {b['win_rate_score']:.1f}/100 (20%)")
        print(f"  Diamond Hands score: {b['dh_score']:.1f}/100 (10%)")


if __name__ == "__main__":
    main()
