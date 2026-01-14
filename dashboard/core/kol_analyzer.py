"""
Advanced KOL Analyzer - Comprehensive metrics and insights

Calculates advanced statistics for each KOL including:
- Trading performance (advanced)
- Temporal patterns
- Behavioral analysis
- Portfolio status
- Trade quality
"""

import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, List, Any
from collections import Counter
import numpy as np

from core.database import db, Trade, ClosedPosition, KOL, OpenPosition
from sqlalchemy import func, and_, case as sql_case
from dashboard.core.data_manager import get_data_manager


class AdvancedKOLAnalyzer:
    """
    Calculate comprehensive advanced metrics for KOLs
    """

    def __init__(self):
        self.data_manager = get_data_manager()

    def analyze_kol(self, session, kol_id: int) -> Dict[str, Any]:
        """
        Get comprehensive analysis of a single KOL

        Args:
            session: Database session
            kol_id: KOL ID

        Returns:
            Dictionary with all advanced metrics
        """
        kol = session.query(KOL).get(kol_id)
        if not kol:
            return {'error': 'KOL not found'}

        # Get all data
        trades = session.query(Trade).filter(Trade.kol_id == kol_id).all()
        closed_positions = session.query(ClosedPosition).filter(ClosedPosition.kol_id == kol_id).all()
        open_positions = session.query(OpenPosition).filter(OpenPosition.kol_id == kol_id).all()

        if not trades:
            return self._empty_analysis(kol)

        # Calculate all advanced metrics
        analysis = {
            'kol_id': kol_id,
            'name': kol.name,
            'wallet_address': kol.wallet_address,

            # 1. Advanced Trading Performance
            'trading_performance': self._calculate_trading_performance(session, closed_positions),

            # 2. Temporal Patterns
            'temporal_patterns': self._calculate_temporal_patterns(trades),

            # 3. Behavioral Analysis
            'behavior': self._calculate_behavioral_patterns(session, kol_id, trades),

            # 4. Current Portfolio
            'portfolio': self._calculate_current_portfolio(open_positions, trades),

            # 5. Trade Quality
            'trade_quality': self._calculate_trade_quality(closed_positions),

            # Metadata
            'first_trade': min(t.timestamp for t in trades),
            'last_trade': max(t.timestamp for t in trades),
            'total_trades': len(trades),
            'analyzed_at': datetime.now().isoformat()
        }

        return analysis

    def _empty_analysis(self, kol: KOL) -> Dict[str, Any]:
        """Return empty analysis for KOL with no trades"""
        return {
            'kol_id': kol.id,
            'name': kol.name,
            'wallet_address': kol.wallet_address,
            'trading_performance': None,
            'temporal_patterns': None,
            'behavior': None,
            'portfolio': None,
            'trade_quality': None,
            'total_trades': 0,
            'analyzed_at': datetime.now().isoformat()
        }

    def _calculate_trading_performance(self, session, closed_positions: List[ClosedPosition]) -> Dict[str, Any]:
        """Calculate advanced trading performance metrics"""
        if not closed_positions:
            return {}

        # Basic stats
        pnls = [pos.pnl_sol for pos in closed_positions]
        multiples = [pos.pnl_multiple for pos in closed_positions]

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        # Biggest win/loss
        biggest_win = max(wins) if wins else 0
        biggest_loss = min(losses) if losses else 0

        # Avg position size (calculate from trades)
        position_sizes = []
        for pos in closed_positions:
            # Find the buy trade for this position
            buy_trade = session.query(Trade).filter(
                and_(
                    Trade.kol_id == pos.kol_id,
                    Trade.token_address == pos.token_address,
                    Trade.operation == 'buy',
                    Trade.timestamp <= pos.exit_time
                )
            ).order_by(Trade.timestamp.desc()).first()

            if buy_trade:
                position_sizes.append(buy_trade.amount_sol)

        avg_position_size = sum(position_sizes) / len(position_sizes) if position_sizes else 0

        # Win/Loss ratio
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        # Profit factor
        total_wins = sum(wins)
        total_losses = abs(sum(losses))
        profit_factor = total_wins / total_losses if total_losses != 0 else 0

        # Sharpe ratio (simplified)
        if len(pnls) > 1:
            avg_pnl = np.mean(pnls)
            std_pnl = np.std(pnls)
            sharpe_ratio = (avg_pnl / std_pnl) if std_pnl != 0 else 0
        else:
            sharpe_ratio = 0

        # Max drawdown
        cumulative_pnl = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdowns = running_max - cumulative_pnl
        max_drawdown = max(drawdowns) if len(drawdowns) > 0 else 0

        # Current streak
        current_streak = 0
        current_streak_type = None
        for i in range(len(closed_positions) - 1, -1, -1):
            if closed_positions[i].pnl_sol > 0:
                if current_streak_type == 'loss' or current_streak_type is None:
                    current_streak += 1
                    current_streak_type = 'win'
                else:
                    break
            else:
                if current_streak_type == 'win' or current_streak_type is None:
                    current_streak += 1
                    current_streak_type = 'loss'
                else:
                    break

        return {
            'biggest_win_sol': round(biggest_win, 2),
            'biggest_loss_sol': round(biggest_loss, 2),
            'avg_position_size_sol': round(avg_position_size, 2),
            'avg_winning_trade_sol': round(avg_win, 2) if wins else 0,
            'avg_losing_trade_sol': round(avg_loss, 2) if losses else 0,
            'win_loss_ratio': round(win_loss_ratio, 2),
            'profit_factor': round(profit_factor, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'max_drawdown_sol': round(max_drawdown, 2),
            'current_streak': current_streak,
            'current_streak_type': current_streak_type
        }

    def _calculate_temporal_patterns(self, trades: List[Trade]) -> Dict[str, Any]:
        """Calculate temporal trading patterns"""
        if not trades:
            return {}

        # Extract temporal data
        hours = [t.timestamp.hour for t in trades]
        weekdays = [t.timestamp.weekday() for t in trades]  # 0=Monday, 6=Sunday

        # Best hour
        hour_counts = Counter(hours)
        best_hour = hour_counts.most_common(1)[0][0] if hour_counts else 0

        # Best weekday
        weekday_counts = Counter(weekdays)
        best_weekday = weekday_counts.most_common(1)[0][0] if weekday_counts else 0

        # Calculate PnL by hour
        hour_pnl = {}
        for trade in trades:
            hour = trade.timestamp.hour
            if trade.operation == 'sell':
                # Find matching buy to calculate PnL (simplified)
                hour_pnl[hour] = hour_pnl.get(hour, 0) + trade.amount_sol

        best_hour_pnl = max(hour_pnl.items(), key=lambda x: x[1]) if hour_pnl else (0, 0)

        # Average time between trades
        timestamps = sorted([t.timestamp for t in trades])
        if len(timestamps) > 1:
            time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() for i in range(len(timestamps)-1)]
            avg_time_between_trades = sum(time_diffs) / len(time_diffs)
        else:
            avg_time_between_trades = 0

        # Trading frequency
        if trades:
            first_trade = min(trades, key=lambda t: t.timestamp).timestamp
            last_trade = max(trades, key=lambda t: t.timestamp).timestamp
            days_trading = (last_trade - first_trade).days + 1
            trades_per_day = len(trades) / days_trading if days_trading > 0 else 0
        else:
            trades_per_day = 0

        return {
            'best_hour': int(best_hour),
            'best_weekday': int(best_weekday),  # 0-6
            'best_hour_for_pnl': int(best_hour_pnl[0]),
            'best_hour_pnl_sol': round(best_hour_pnl[1], 2),
            'avg_time_between_trades_minutes': round(avg_time_between_trades / 60, 2),
            'trades_per_day': round(trades_per_day, 2),
            'most_active_hour': int(hour_counts.most_common(1)[0][0] if hour_counts else 0)
        }

    def _calculate_behavioral_patterns(self, session, kol_id: int, trades: List[Trade]) -> Dict[str, Any]:
        """Calculate behavioral patterns"""
        if not trades:
            return {}

        # Favorite tokens
        token_counts = Counter(t.token_address for t in trades)
        top_tokens = token_counts.most_common(5)

        # DEX preference
        dex_counts = Counter(t.dex for t in trades if t.dex)

        # Calculate buy & hold vs quick flip
        buys_sells = {}
        for token in set(t.token_address for t in trades):
            token_trades = [t for t in trades if t.token_address == token]
            buys = sum(1 for t in token_trades if t.operation == 'buy')
            sells = sum(1 for t in token_trades if t.operation == 'sell')
            if buys > 0:
                buys_sells[token] = sells / buys

        avg_flip_ratio = sum(buys_sells.values()) / len(buys_sells) if buys_sells else 0

        # FOMO score (buys near peaks) - simplified
        # Count how often they buy when price is about to drop
        fomo_score = 0  # TODO: implement with price data

        # Check if they follow other KOLs (correlation)
        # For now, use a simple metric: do they trade the same tokens as top performers?
        correlation_score = 0  # TODO: implement

        # Buy vs Sell ratio
        total_buys = sum(1 for t in trades if t.operation == 'buy')
        total_sells = sum(1 for t in trades if t.operation == 'sell')
        buy_sell_ratio = total_buys / total_sells if total_sells > 0 else 0

        return {
            'favorite_tokens': [
                {'address': token[:10] + '...', 'count': count}
                for token, count in top_tokens
            ],
            'dex_preference': dict(dex_counts),
            'avg_flip_ratio': round(avg_flip_ratio, 2),
            'buy_sell_ratio': round(buy_sell_ratio, 2),
            'fomo_score': round(fomo_score, 2),  # 0-100
            'correlation_with_top_kols': round(correlation_score, 2)  # 0-100
        }

    def _calculate_current_portfolio(self, open_positions: List[OpenPosition], trades: List[Trade]) -> Dict[str, Any]:
        """Calculate current portfolio status"""
        if not open_positions:
            return {
                'has_open_positions': False,
                'total_value_sol': 0,
                'total_unrealized_pnl_sol': 0,
                'positions': []
            }

        positions = []
        total_value = 0
        total_unrealized_pnl = 0

        for pos in open_positions:
            value = pos.current_value_sol
            pnl = pos.unrealized_pnl_sol

            positions.append({
                'token_address': pos.token_address[:10] + '...',
                'entry_time': pos.entry_time.isoformat(),
                'entry_price': round(pos.entry_price, 8),
                'current_price': round(pos.current_price, 8) if pos.current_price else None,
                'amount_sol': round(pos.entry_amount_sol, 2),
                'current_value_sol': round(value, 2),
                'unrealized_pnl_sol': round(pnl, 2),
                'unrealized_pnl_multiple': round(pos.unrealized_pnl_multiple, 2),
                'hold_time_hours': round(pos.hold_time_hours, 2),
                'peak_multiple_reached': round(pos.peak_multiple_reached, 2)
            })

            total_value += value
            total_unrealized_pnl += pnl

        # Calculate concentration
        if total_value > 0:
            concentrations = [(p['current_value_sol'] / total_value * 100) for p in positions]
            top3_concentration = round(sum(sorted(concentrations, reverse=True)[:3]), 2)
        else:
            top3_concentration = 0

        return {
            'has_open_positions': True,
            'total_value_sol': round(total_value, 2),
            'total_unrealized_pnl_sol': round(total_unrealized_pnl, 2),
            'num_positions': len(positions),
            'top3_concentration_percentage': top3_concentration,
            'positions': positions
        }

    def _calculate_trade_quality(self, closed_positions: List[ClosedPosition]) -> Dict[str, Any]:
        """Calculate trade quality metrics"""
        if not closed_positions:
            return {}

        # Average hold time
        hold_times = [pos.hold_time_seconds for pos in closed_positions]
        avg_hold_time = sum(hold_times) / len(hold_times) if hold_times else 0

        # Sold too early vs held too long
        # This would require price history data, simplified for now
        sold_early_count = 0  # TODO: implement with TradeQuality model
        held_long_count = 0

        # Timing score (0-100)
        # Based on win rate and avg hold time
        win_rate = sum(1 for pos in closed_positions if pos.pnl_sol > 0) / len(closed_positions)
        timing_score = min(100, win_rate * 100)  # Simplified

        # Best and worst trades
        trades_by_pnl = sorted(closed_positions, key=lambda x: x.pnl_sol, reverse=True)
        best_trade = trades_by_pnl[0] if trades_by_pnl else None
        worst_trade = trades_by_pnl[-1] if trades_by_pnl else None

        return {
            'avg_hold_time_hours': round(avg_hold_time / 3600, 2),
            'sold_too_early_rate': round(sold_early_count / len(closed_positions) * 100, 2) if closed_positions else 0,
            'held_too_long_rate': round(held_long_count / len(closed_positions) * 100, 2) if closed_positions else 0,
            'timing_score': round(timing_score, 2),
            'best_trade': {
                'token': best_trade.token_address[:10] + '...' if best_trade else None,
                'pnl_sol': round(best_trade.pnl_sol, 2) if best_trade else 0,
                'multiple': round(best_trade.pnl_multiple, 2) if best_trade else 0
            } if best_trade else None,
            'worst_trade': {
                'token': worst_trade.token_address[:10] + '...' if worst_trade else None,
                'pnl_sol': round(worst_trade.pnl_sol, 2) if worst_trade else 0,
                'multiple': round(worst_trade.pnl_multiple, 2) if worst_trade else 0
            } if worst_trade else None
        }

    def get_all_kols_advanced_analysis(self, session) -> List[Dict[str, Any]]:
        """
        Get advanced analysis for all KOLs

        Returns:
            List of KOL analyses
        """
        kols = session.query(KOL).all()
        analyses = []

        for kol in kols:
            analysis = self.analyze_kol(session, kol.id)
            analyses.append(analysis)

        return analyses

    def get_kol_summary_dict(self, session, kol_id: int) -> Dict[str, Any]:
        """
        Get a summary dictionary for dashboard display
        """
        analysis = self.analyze_kol(session, kol_id)

        if 'error' in analysis:
            return {'error': analysis['error']}

        # Create summary for display
        summary = {
            'kol_id': analysis['kol_id'],
            'name': analysis['name'],

            # Quick stats
            'total_trades': analysis['total_trades'],
            'last_trade': analysis['last_trade'],

            # Performance
            'biggest_win': analysis['trading_performance']['biggest_win_sol'] if analysis.get('trading_performance') else 0,
            'biggest_loss': analysis['trading_performance']['biggest_loss_sol'] if analysis.get('trading_performance') else 0,
            'sharpe_ratio': analysis['trading_performance']['sharpe_ratio'] if analysis.get('trading_performance') else 0,
            'max_drawdown': analysis['trading_performance']['max_drawdown_sol'] if analysis.get('trading_performance') else 0,
            'current_streak': analysis['trading_performance']['current_streak'] if analysis.get('trading_performance') else 0,
            'current_streak_type': analysis['trading_performance']['current_streak_type'] if analysis.get('trading_performance') else None,

            # Temporal
            'trades_per_day': analysis['temporal_patterns']['trades_per_day'] if analysis.get('temporal_patterns') else 0,
            'best_hour': analysis['temporal_patterns']['best_hour'] if analysis.get('temporal_patterns') else 0,

            # Behavior
            'favorite_tokens': analysis['behavior']['favorite_tokens'][:3] if analysis.get('behavior') else [],
            'buy_sell_ratio': analysis['behavior']['buy_sell_ratio'] if analysis.get('behavior') else 0,

            # Portfolio
            'has_open_positions': analysis['portfolio']['has_open_positions'] if analysis.get('portfolio') else False,
            'portfolio_value': analysis['portfolio']['total_value_sol'] if analysis.get('portfolio') else 0,
            'unrealized_pnl': analysis['portfolio']['total_unrealized_pnl_sol'] if analysis.get('portfolio') else 0,

            # Quality
            'timing_score': analysis['trade_quality']['timing_score'] if analysis.get('trade_quality') else 0,
            'best_trade_pnl': analysis['trade_quality']['best_trade']['pnl_sol'] if analysis.get('trade_quality') and analysis['trade_quality'].get('best_trade') else 0,

            'full_analysis': analysis
        }

        return summary


# Singleton instance
_analyzer = None


def get_advanced_analyzer() -> AdvancedKOLAnalyzer:
    """Get singleton instance"""
    global _analyzer
    if _analyzer is None:
        _analyzer = AdvancedKOLAnalyzer()
    return _analyzer
