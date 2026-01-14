"""
Enhanced Summary Generator - Comprehensive Run Summaries

Generates detailed summaries every 15 minutes including:
- Run performance metrics
- Period-specific top performers (last 15min, NOT historical)
- KOL trading statistics
- Token analysis
- DEX analysis
- Market insights
- Trends and comparisons
"""

import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json
from pathlib import Path

from core.database import db, Trade, ClosedPosition, KOL
from sqlalchemy import func, case as sql_case, and_
from dashboard.core.data_manager import get_data_manager


class SummaryGenerator:
    """Generate comprehensive automated summaries of tracker performance"""

    def __init__(self):
        self.data_manager = get_data_manager()
        self.summaries_dir = Path('data/summaries')
        self.summaries_dir.mkdir(exist_ok=True)

    def generate_summary(self) -> Dict[str, Any]:
        """
        Generate comprehensive summary

        Returns:
            Dictionary with summary data
        """
        session = db.get_session()

        try:
            now = datetime.now()
            fifteen_min_ago = now - timedelta(minutes=15)

            # Get all statistics
            run_stats = self._get_run_stats(session, fifteen_min_ago)
            kol_stats = self._get_kol_stats(session, fifteen_min_ago)
            trading_stats = self._get_trading_stats(session, fifteen_min_ago)

            # Period-specific (NEW data from last 15 min)
            period_performers = self._get_period_performers(session, fifteen_min_ago)
            period_tokens = self._get_period_tokens(session, fifteen_min_ago)
            period_dexs = self._get_period_dex_stats(session, fifteen_min_ago)

            # All-time top performers
            all_time_performers = self._get_all_time_performers(session)

            # Recent activity
            recent_activity = self._get_recent_activity(session)

            # Trends and insights
            trends = self._get_trends(session)

            # Compile summary
            summary = {
                'timestamp': now.isoformat(),
                'period_start': fifteen_min_ago.isoformat(),
                'run_stats': run_stats,
                'kol_stats': kol_stats,
                'trading_stats': trading_stats,
                'period_performers': period_performers,  # Top performers in last 15min
                'all_time_performers': all_time_performers,  # Historical best
                'period_tokens': period_tokens,  # Most traded tokens
                'period_dexs': period_dexs,  # DEX usage
                'recent_activity': recent_activity,
                'trends': trends
            }

            return summary

        finally:
            session.close()

    def _get_run_stats(self, session, period_start: datetime) -> Dict[str, Any]:
        """Get run/tracker statistics"""
        total_kols = session.query(KOL).count()
        total_trades = session.query(Trade).count()
        total_positions = session.query(ClosedPosition).count()

        # Recent activity
        recent_trades = session.query(Trade).filter(
            Trade.timestamp >= period_start
        ).count()

        # KOLs active in this period
        active_kols = session.query(Trade.kol_id).filter(
            Trade.timestamp >= period_start
        ).distinct().count()

        return {
            'total_kols': total_kols,
            'total_trades': total_trades,
            'total_closed_positions': total_positions,
            'recent_trades_last_15min': recent_trades,
            'active_kols_period': active_kols,
            'coverage_percentage': (total_kols / 450) * 100 if total_kols > 0 else 0
        }

    def _get_kol_stats(self, session, period_start: datetime) -> Dict[str, Any]:
        """Get KOL performance statistics"""
        df = self.data_manager.load_leaderboard()

        if df.empty:
            return {'message': 'No leaderboard data available'}

        # Calculate statistics
        avg_score = df['diamond_hand_score'].mean()
        avg_win_rate = (df['win_rate'] * 100).mean()
        avg_hold_time = df['avg_hold_time_hours'].mean()

        # Count by type
        diamond_hands = df['is_diamond_hand'].sum()
        scalpers = df['is_scalper'].sum()

        # NEW: Period-specific KOL stats (last 15 min)
        period_trades = session.query(Trade).filter(
            Trade.timestamp >= period_start
        ).count()

        period_kols = session.query(func.count(func.distinct(Trade.kol_id))).filter(
            Trade.timestamp >= period_start
        ).scalar()

        # Profitable positions in period
        period_pnl = session.query(
            func.sum(ClosedPosition.pnl_sol)
        ).filter(
            ClosedPosition.exit_time >= period_start
        ).scalar() or 0

        return {
            'avg_diamond_hand_score': round(avg_score, 2),
            'avg_win_rate_percentage': round(avg_win_rate, 2),
            'avg_hold_time_hours': round(avg_hold_time, 2),
            'total_diamond_hands': int(diamond_hands),
            'total_scalpers': int(scalpers),
            'total_analyzed': len(df),
            'period_avg_trades_per_kol': round(period_trades / period_kols, 2) if period_kols > 0 else 0,
            'period_total_pnl_sol': round(period_pnl, 2)
        }

    def _get_trading_stats(self, session, period_start: datetime) -> Dict[str, Any]:
        """Get trading activity statistics"""
        # All-time volume
        total_volume = session.query(
            func.sum(Trade.amount_sol)
        ).scalar() or 0

        # Period volume (last 15 min)
        period_volume = session.query(
            func.sum(Trade.amount_sol)
        ).filter(
            Trade.timestamp >= period_start
        ).scalar() or 0

        # By operation (period)
        period_buys = session.query(Trade).filter(
            and_(Trade.operation == 'buy', Trade.timestamp >= period_start)
        ).count()

        period_sells = session.query(Trade).filter(
            and_(Trade.operation == 'sell', Trade.timestamp >= period_start)
        ).count()

        # All-time profitability
        profitable_positions = session.query(ClosedPosition).filter(
            ClosedPosition.pnl_sol > 0
        ).count()
        total_positions = session.query(ClosedPosition).count()
        all_time_win_rate = (profitable_positions / total_positions * 100) if total_positions > 0 else 0

        # Period profitability
        period_profitable = session.query(ClosedPosition).filter(
            and_(ClosedPosition.pnl_sol > 0, ClosedPosition.exit_time >= period_start)
        ).count()

        period_total_pos = session.query(ClosedPosition).filter(
            ClosedPosition.exit_time >= period_start
        ).count()

        period_win_rate = (period_profitable / period_total_pos * 100) if period_total_pos > 0 else 0

        # By DEX
        dex_count = session.query(
            func.count(func.distinct(Trade.dex))
        ).scalar()

        return {
            'total_volume_sol': round(total_volume, 2),
            'period_volume_sol': round(period_volume, 2),
            'total_buys': session.query(Trade).filter(Trade.operation == 'buy').count(),
            'total_sells': session.query(Trade).filter(Trade.operation == 'sell').count(),
            'period_buys': period_buys,
            'period_sells': period_sells,
            'buy_sell_ratio': round(period_buys / period_sells, 2) if period_sells > 0 else 0,
            'overall_win_rate_percentage': round(all_time_win_rate, 2),
            'period_win_rate_percentage': round(period_win_rate, 2),
            'active_dexs': int(dex_count)
        }

    def _get_period_performers(self, session, period_start: datetime) -> Dict[str, Any]:
        """
        Get top performers IN THIS PERIOD (last 15 min)
        This shows who's trading NOW, not historical
        """
        # Get KOLs with most trades in this period
        from sqlalchemy import case as sql_case

        kol_trade_counts = session.query(
            KOL.name,
            KOL.id,
            func.count(Trade.id).label('trade_count'),
            func.sum(
                sql_case((Trade.operation == 'buy', Trade.amount_sol), else_=0)
            ).label('bought_sol'),
            func.sum(
                sql_case((Trade.operation == 'sell', Trade.amount_sol), else_=0)
            ).label('sold_sol')
        ).join(
            Trade, KOL.id == Trade.kol_id
        ).filter(
            Trade.timestamp >= period_start
        ).group_by(
            KOL.id, KOL.name
        ).order_by(
            func.count(Trade.id).desc()
        ).limit(10).all()

        period_top = []
        for kol_name, kol_id, trade_count, bought, sold in kol_trade_counts:
            # Get their stats
            leaderboard = self.data_manager.load_leaderboard()
            kol_data = leaderboard[leaderboard['kol_id'] == kol_id]

            if not kol_data.empty:
                score = kol_data['diamond_hand_score'].iloc[0]
                win_rate = kol_data['win'].iloc[0] if 'win' in kol_data.columns else 0
            else:
                score = 0
                win_rate = 0

            period_top.append({
                'name': kol_name,
                'kol_id': kol_id,
                'period_trades': int(trade_count),
                'period_bought_sol': round(float(bought or 0), 2),
                'period_sold_sol': round(float(sold or 0), 2),
                'net_volume_sol': round(float(sold or 0) - float(bought or 0), 2),
                'diamond_hand_score': round(float(score), 2),
                'win_rate': round(float(win_rate), 2)
            })

        return {
            'top_10_active': period_top,
            'most_active_name': period_top[0]['name'] if period_top else None,
            'most_active_trades': period_top[0]['period_trades'] if period_top else 0
        }

    def _get_all_time_performers(self, session) -> Dict[str, Any]:
        """Get all-time top performers"""
        df = self.data_manager.load_leaderboard()

        if df.empty:
            return {'top_10': []}

        top_10 = df.nlargest(10, 'diamond_hand_score')[[
            'rank', 'name', 'diamond_hand_score', 'total_trades',
            'win_rate', 'total_pnl_sol', 'avg_hold_time_hours'
        ]].head(10)

        return {
            'top_10': top_10.to_dict('records'),
            'best_score': float(top_10['diamond_hand_score'].iloc[0]),
            'best_kol': top_10['name'].iloc[0]
        }

    def _get_period_tokens(self, session, period_start: datetime) -> List[Dict[str, Any]]:
        """Get most traded tokens in this period"""
        from sqlalchemy import case as sql_case

        token_stats = session.query(
            Trade.token_address,
            func.count(Trade.id).label('trade_count'),
            func.count(func.distinct(Trade.kol_id)).label('unique_kols'),
            func.sum(
                sql_case((Trade.operation == 'buy', Trade.amount_sol), else_=0)
            ).label('total_bought')
        ).filter(
            Trade.timestamp >= period_start
        ).group_by(
            Trade.token_address
        ).order_by(
            func.count(Trade.id).desc()
        ).limit(10).all()

        tokens = []
        for token_addr, trade_count, unique_kols, total_bought in token_stats:
            tokens.append({
                'token_address': token_addr[:10] + '...',
                'full_address': token_addr,
                'trade_count': int(trade_count),
                'unique_kols_trading': int(unique_kols),
                'total_volume_sol': round(float(total_bought or 0), 2)
            })

        return tokens

    def _get_period_dex_stats(self, session, period_start: datetime) -> List[Dict[str, Any]]:
        """Get DEX usage statistics for this period"""
        dex_stats = session.query(
            Trade.dex,
            func.count(Trade.id).label('trade_count'),
            func.sum(Trade.amount_sol).label('volume')
        ).filter(
            Trade.timestamp >= period_start
        ).group_by(
            Trade.dex
        ).order_by(
            func.count(Trade.id).desc()
        ).all()

        dexs = []
        for dex_name, trade_count, volume in dex_stats:
            dexs.append({
                'name': dex_name or 'Unknown',
                'trade_count': int(trade_count),
                'volume_sol': round(float(volume or 0), 2)
            })

        return dexs

    def _get_recent_activity(self, session) -> List[Dict[str, Any]]:
        """Get recent trading activity (last 10 trades)"""
        recent_trades = session.query(Trade).order_by(
            Trade.timestamp.desc()
        ).limit(20).all()

        activity = []
        for trade in recent_trades:
            kol = session.query(KOL).filter(
                KOL.id == trade.kol_id
            ).first()

            activity.append({
                'kol_name': kol.name if kol else 'Unknown',
                'operation': trade.operation,
                'token': trade.token_address[:10] + '...',
                'amount_sol': round(trade.amount_sol, 4),
                'dex': trade.dex or 'Unknown',
                'timestamp': trade.timestamp.isoformat()
            })

        return activity

    def _get_trends(self, session) -> Dict[str, Any]:
        """Get trends and insights"""
        # Compare last 2 periods (30min total)
        now = datetime.now()
        period1_start = now - timedelta(minutes=15)
        period2_start = now - timedelta(minutes=30)

        period1_trades = session.query(Trade).filter(
            Trade.timestamp >= period1_start
        ).count()

        period2_trades = session.query(Trade).filter(
            and_(
                Trade.timestamp >= period2_start,
                Trade.timestamp < period1_start
            )
        ).count()

        trend_direction = "up" if period1_trades > period2_trades else "down" if period1_trades < period2_trades else "stable"
        trend_percentage = ((period1_trades - period2_trades) / period2_trades * 100) if period2_trades > 0 else 0

        return {
            'current_period_trades': period1_trades,
            'previous_period_trades': period2_trades,
            'trend_direction': trend_direction,
            'trend_percentage': round(trend_percentage, 1)
        }

    def save_summary(self, summary: Dict[str, Any]) -> str:
        """Save summary to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'summary_{timestamp}.json'
        filepath = self.summaries_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, default=str)

        return str(filepath)

    def get_summary_markdown(self, summary: Dict[str, Any]) -> str:
        """Convert summary to markdown format"""
        md = f"""
# 📊 KOL Tracker - Comprehensive Run Summary

**Generated:** {summary['timestamp']}
**Period:** {summary['period_start']} → {summary['timestamp']}

---

## 🎯 System Status

| Metric | Value |
|--------|-------|
| **Total KOLs Tracked** | {summary['run_stats']['total_kols']:,} |
| **Total Trades (All Time)** | {summary['run_stats']['total_trades']:,} |
| **Closed Positions** | {summary['run_stats']['total_closed_positions']:,} |
| **Recent Trades (15min)** | {summary['run_stats']['recent_trades_last_15min']:,} |
| **Active KOLs (Period)** | {summary['run_stats']['active_kols_period']:,} |
| **Coverage** | {summary['run_stats']['coverage_percentage']:.1f}% |

---

## 📈 Period Trends (Last 15min vs Previous 15min)

| Metric | Current | Previous | Trend |
|--------|---------|----------|-------|
| **Trades** | {summary['trends']['current_period_trades']:,} | {summary['trends']['previous_period_trades']:,} |
| **Direction** | | | {summary['trends']['trend_direction'].upper()} ({summary['trends']['trend_percentage']:+.1f}%) |

---

## 💹 Trading Activity

### All-Time Stats
| Metric | Value |
|--------|-------|
| **Total Volume (SOL)** | {summary['trading_stats']['total_volume_sol']:,.2f} |
| **Total Buys** | {summary['trading_stats']['total_buys']:,} |
| **Total Sells** | {summary['trading_stats']['total_sells']:,} |
| **Overall Win Rate** | {summary['trading_stats']['overall_win_rate_percentage']:.2f}% |
| **Active DEXs** | {summary['trading_stats']['active_dexs']} |

### This Period (Last 15min)
| Metric | Value |
|--------|-------|
| **Period Volume** | {summary['trading_stats']['period_volume_sol']:,.2f} SOL |
| **Buys** | {summary['trading_stats']['period_buys']:,} |
| **Sells** | {summary['trading_stats']['period_sells']:,} |
| **Buy/Sell Ratio** | {summary['trading_stats']['buy_sell_ratio']:.2f} |
| **Period Win Rate** | {summary['trading_stats']['period_win_rate_percentage']:.2f}% |

---

## 👥 KOL Performance Analysis

| Metric | Value |
|--------|-------|
| **Avg Diamond Hand Score** | {summary['kol_stats']['avg_diamond_hand_score']} |
| **Avg Win Rate** | {summary['kol_stats']['avg_win_rate_percentage']:.2f}% |
| **Avg Hold Time** | {summary['kol_stats']['avg_hold_time_hours']:.2f} hours |
| **Diamond Hands** | {summary['kol_stats']['total_diamond_hands']} |
| **Scalpers** | {summary['kol_stats']['total_scalpers']} |
| **KOLs Analyzed** | {summary['kol_stats']['total_analyzed']} |
| **Period PnL** | {summary['kol_stats']['period_total_pnl_sol']:+.2f} SOL |

---

## 🔥 Most Active KOLs (This Period)

*KOLs with the most trading activity in the last 15 minutes*

| Rank | KOL | Trades | Bought | Sold | Net Flow | Score | Win Rate |
|------|-----|--------|--------|------|----------|-------|----------|
"""

        # Most active in this period
        for i, perf in enumerate(summary['period_performers']['top_10_active'][:10], 1):
            flow_emoji = "📈" if perf['net_volume_sol'] > 0 else "📉" if perf['net_volume_sol'] < 0 else "➡️"
            md += f"| {i} | **{perf['name']}** | {perf['period_trades']} | {perf['period_bought_sol']:.2f} | {perf['period_sold_sol']:.2f} | {flow_emoji} {perf['net_volume_sol']:+.2f} | {perf['diamond_hand_score']:.1f} | {perf['win_rate']*100:.1f}% |\n"

        md += f"""
---

## 🏆 All-Time Top Performers

*Best performers across all time*

| Rank | KOL | Score | Trades | Win Rate | PnL (SOL) | Avg Hold |
|------|-----|-------|--------|----------|-----------|----------|
"""

        for performer in summary['all_time_performers']['top_10']:
            md += f"| {performer['rank']} | **{performer['name']}** | {performer['diamond_hand_score']:.1f} | {int(performer['total_trades'])} | {performer['win_rate']*100:.1f}% | {performer['total_pnl_sol']:.2f} | {performer['avg_hold_time_hours']:.2f}h |\n"

        md += f"""
---

## 🪙 Top Tokens (This Period)

*Most traded tokens in the last 15 minutes*

| Rank | Token | Trades | Unique KOLs | Volume (SOL) |
|------|-------|--------|-------------|--------------|
"""

        for i, token in enumerate(summary['period_tokens'][:10], 1):
            md += f"| {i} | `{token['token_address']}` | {token['trade_count']} | {token['unique_kols_trading']} | {token['total_volume_sol']:.2f} |\n"

        md += f"""
---

## 🏛️ DEX Usage (This Period)

*Trading activity by DEX in the last 15 minutes*

| DEX | Trades | Volume (SOL) | Share |
|-----|--------|--------------|-------|
"""

        total_period_trades = sum(d['trade_count'] for d in summary['period_dexs'])
        for dex in summary['period_dexs']:
            share = (dex['trade_count'] / total_period_trades * 100) if total_period_trades > 0 else 0
            md += f"| **{dex['name']}** | {dex['trade_count']:,} | {dex['volume_sol']:,.2f} | {share:.1f}% |\n"

        md += f"""
---

## 🔄 Recent Activity

*Last 20 trades across all KOLs*

"""

        for activity in summary['recent_activity'][:20]:
            emoji = "🟢" if activity['operation'] == 'buy' else "🔴"
            op = activity['operation'].upper()
            md += f"- {emoji} **{activity['kol_name']}**: {op} {activity['amount_sol']} SOL of `{activity['token']}` on {activity['dex']}\n"

        md += f"""
---

## 📊 Key Insights

1. **Most Active**: {summary['period_performers']['most_active_name']} with {summary['period_performers']['most_active_trades']} trades this period
2. **Trend**: Activity is {summary['trends']['trend_direction']} ({summary['trends']['trend_percentage']:+.1f}% from previous period)
3. **Period Win Rate**: {summary['trading_stats']['period_win_rate_percentage']:.1f}% of positions profitable
4. {f"**Top Token**: `{summary['period_tokens'][0]['token_address']}` with {summary['period_tokens'][0]['trade_count']} trades" if summary['period_tokens'] else "**Top Token**: No activity"}
5. {f"**Top DEX**: {summary['period_dexs'][0]['name']} with {summary['period_dexs'][0]['trade_count']} trades" if summary['period_dexs'] else "**Top DEX**: No activity"}

---

*Generated automatically by KOL Tracker ML*
*Next summary in 15 minutes*
"""

        return md


# Singleton instance
_summary_generator = None


def get_summary_generator() -> SummaryGenerator:
    """Get singleton instance of summary generator"""
    global _summary_generator
    if _summary_generator is None:
        _summary_generator = SummaryGenerator()
    return _summary_generator
