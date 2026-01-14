"""
Data Manager for Dashboard

Handles all database queries with caching and optimization.
Provides efficient data loading for dashboard components.
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

from core.database import db, KOL, Trade, ClosedPosition
from dashboard.core.cache_manager import streamlit_cached
from dashboard.core.state_manager import get_state


class DataManager:
    """
    Manages data loading from database with intelligent caching.

    Features:
    - Cached queries for common operations
    - Lazy loading of large datasets
    - Filter-based query optimization
    - Pagination support
    """

    def __init__(self):
        """Initialize data manager"""
        self.state = get_state()

    @streamlit_cached(ttl=60)  # Cache for 1 minute
    def get_database_stats(_self) -> Dict[str, int]:
        """
        Get overall database statistics

        Returns:
            Dictionary with counts
        """
        session = db.get_session()

        try:
            stats = {
                'total_kols': session.query(KOL).count(),
                'total_trades': session.query(Trade).count(),
                'total_positions': session.query(ClosedPosition).count(),
            }

            # Count KOLs with trades
            from sqlalchemy import func
            stats['kols_with_trades'] = session.query(
                func.count(func.distinct(Trade.kol_id))
            ).scalar()

            return stats
        finally:
            session.close()

    @streamlit_cached(ttl=300)  # Cache for 5 minutes
    def load_leaderboard(_self) -> pd.DataFrame:
        """
        Load leaderboard data from JSON file

        Returns:
            DataFrame with leaderboard data
        """
        try:
            import json
            from pathlib import Path

            leaderboard_file = Path('data/leaderboard.json')
            if not leaderboard_file.exists():
                return pd.DataFrame()

            with open(leaderboard_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            df = pd.DataFrame(data['leaderboard'])

            # Ensure proper data types to avoid PyArrow errors
            if not df.empty:
                # Convert all numeric columns to proper types
                numeric_cols = ['kol_id', 'total_trades', 'avg_hold_time_hours',
                               'median_hold_time_hours', 'hold_time_std_hours',
                               'total_pnl_sol', 'win_rate', 'avg_multiple',
                               'three_x_plus_count', 'three_x_plus_rate',
                               'three_x_avg_hold_hours', 'consistency_score',
                               'diamond_hand_score', 'rank']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                # Ensure string columns are strings (critical for PyArrow)
                string_cols = ['name', 'wallet_address']
                for col in string_cols:
                    if col in df.columns:
                        df[col] = df[col].astype(str).str.replace('nan', '').replace('None', '')

                # Ensure boolean columns are bool
                bool_cols = ['is_diamond_hand', 'is_scalper']
                for col in bool_cols:
                    if col in df.columns:
                        df[col] = df[col].astype(bool)

                # CRITICAL: Make a deep copy to avoid any reference issues
                df = df.copy()

            return df
        except Exception as e:
            st.error(f"Error loading leaderboard: {e}")
            return pd.DataFrame()

    @streamlit_cached(ttl=60)
    def get_recent_trades(_self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get recent trades with KOL data

        Args:
            limit: Maximum number of trades to return

        Returns:
            List of trade dictionaries
        """
        session = db.get_session()

        try:
            from sqlalchemy.orm import selectinload

            trades = session.query(Trade).options(
                selectinload(Trade.kol)
            ).order_by(
                Trade.timestamp.desc()
            ).limit(limit).all()

            # Convert to dict to avoid detached instance issues
            trades_data = []
            for trade in trades:
                trades_data.append({
                    'kol_id': trade.kol_id,
                    'kol_name': trade.kol.name if trade.kol else 'Unknown',
                    'kol_wallet': trade.kol.wallet_address if trade.kol else 'Unknown',
                    'operation': trade.operation,
                    'token_address': trade.token_address,
                    'amount_sol': trade.amount_sol,
                    'amount_token': trade.amount_token,
                    'dex': trade.dex,
                    'timestamp': trade.timestamp
                })

            return trades_data
        finally:
            session.close()

    def get_filtered_kols(
        self,
        min_score: float = 0,
        min_trades: int = 1,
        max_trades: int = 10000,
        win_rate_range: Tuple[float, float] = (0.0, 1.0),
        kol_type: str = 'all',
        search_query: str = ''
    ) -> pd.DataFrame:
        """
        Get KOLs with filters applied

        Args:
            min_score: Minimum diamond hand score
            min_trades: Minimum number of trades
            max_trades: Maximum number of trades
            win_rate_range: Tuple of (min, max) win rate
            kol_type: Filter by type (all, diamond_hands, scalpers)
            search_query: Search by name or wallet

        Returns:
            Filtered DataFrame of KOLs
        """
        df = self.load_leaderboard()

        if df.empty:
            return df

        # Apply filters
        filtered = df[df['diamond_hand_score'] >= min_score].copy()
        filtered = filtered[filtered['total_trades'] >= min_trades]
        filtered = filtered[filtered['total_trades'] <= max_trades]
        filtered = filtered[
            (filtered['win_rate'] >= win_rate_range[0]) &
            (filtered['win_rate'] <= win_rate_range[1])
        ]

        # Filter by type
        if kol_type == 'diamond_hands_only':
            filtered = filtered[filtered['is_diamond_hand'] == True]
        elif kol_type == 'scalpers_only':
            filtered = filtered[filtered['is_scalper'] == True]

        # Search query
        if search_query:
            search_lower = search_query.lower()
            mask = (
                filtered['name'].str.lower().str.contains(search_lower, na=False) |
                filtered['wallet_address'].str.lower().str.contains(search_lower, na=False)
            )
            filtered = filtered[mask]

        return filtered

    @streamlit_cached(ttl=300)
    def get_kol_details(_self, kol_id: int) -> Optional[Dict[str, Any]]:
        """
        Get detailed information for a single KOL

        Args:
            kol_id: KOL database ID

        Returns:
            Dictionary with KOL details or None
        """
        session = db.get_session()

        try:
            # Convert kol_id to native int to avoid numpy.int64 PostgreSQL error
            kol_id = int(kol_id)
            kol = session.query(KOL).filter(KOL.id == kol_id).first()

            if not kol:
                return None

            return {
                'id': kol.id,
                'name': kol.name,
                'wallet_address': kol.wallet_address,
                'twitter': kol.twitter,
                'telegram': kol.telegram,
                'bio': kol.bio
            }
        finally:
            session.close()

    def get_kol_positions(
        self,
        kol_id: int,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get closed positions for a KOL

        Args:
            kol_id: KOL database ID
            limit: Maximum positions to return

        Returns:
            List of position dictionaries
        """
        session = db.get_session()

        try:
            # Convert kol_id to native int to avoid numpy.int64 PostgreSQL error
            kol_id = int(kol_id)

            positions = session.query(ClosedPosition).filter(
                ClosedPosition.kol_id == kol_id
            ).order_by(
                ClosedPosition.exit_time.desc()
            ).limit(limit).all()

            positions_data = []
            for pos in positions:
                positions_data.append({
                    'token_address': pos.token_address,
                    'entry_time': pos.entry_time,
                    'exit_time': pos.exit_time,
                    'hold_time_hours': pos.hold_time_hours,
                    'pnl_sol': pos.pnl_sol,
                    'pnl_multiple': pos.pnl_multiple,
                    'is_profitable': pos.is_profitable,
                    'is_diamond_hand': pos.is_diamond_hand
                })

            return positions_data
        finally:
            session.close()

    @streamlit_cached(ttl=60)
    def get_token_analysis(_self) -> pd.DataFrame:
        """
        Get aggregated data by token

        Returns:
            DataFrame with token statistics
        """
        session = db.get_session()

        try:
            from sqlalchemy import func, case

            # Query aggregated token stats
            query = session.query(
                ClosedPosition.token_address,
                func.count(ClosedPosition.id).label('trade_count'),
                func.count(func.distinct(ClosedPosition.kol_id)).label('unique_kols'),
                func.avg(ClosedPosition.pnl_multiple).label('avg_multiple'),
                func.sum(ClosedPosition.pnl_sol).label('total_pnl'),
                func.sum(
                    case(
                        (ClosedPosition.pnl_multiple >= 3.0, 1),
                        else_=0
                    )
                ).label('three_x_count')
            ).group_by(ClosedPosition.token_address)

            df = pd.read_sql(query.statement, session.bind)
            return df
        finally:
            session.close()

    @streamlit_cached(ttl=300)
    def get_dex_distribution(_self) -> pd.DataFrame:
        """
        Get distribution of trades by DEX

        Returns:
            DataFrame with DEX statistics
        """
        session = db.get_session()

        try:
            from sqlalchemy import func

            query = session.query(
                Trade.dex,
                func.count(Trade.id).label('trade_count'),
                func.sum(Trade.amount_sol).label('total_volume')
            ).group_by(Trade.dex)

            df = pd.read_sql(query.statement, session.bind)
            return df
        finally:
            session.close()

    def get_time_series_data(
        self,
        granularity: str = 'day',
        days_back: int = 30
    ) -> pd.DataFrame:
        """
        Get time series data for analytics

        Args:
            granularity: 'day', 'week', or 'month'
            days_back: Number of days to look back

        Returns:
            DataFrame with time series data
        """
        session = db.get_session()

        try:
            from sqlalchemy import func, extract

            start_date = datetime.now() - timedelta(days=days_back)

            # Determine extract function based on granularity
            if granularity == 'day':
                date_part = extract('day', ClosedPosition.exit_time)
                month_part = extract('month', ClosedPosition.exit_time)
                year_part = extract('year', ClosedPosition.exit_time)
            elif granularity == 'week':
                # Week extraction is more complex, simplify to day for now
                date_part = extract('day', ClosedPosition.exit_time)
                month_part = extract('month', ClosedPosition.exit_time)
                year_part = extract('year', ClosedPosition.exit_time)
            else:  # month
                month_part = extract('month', ClosedPosition.exit_time)
                year_part = extract('year', ClosedPosition.exit_time)
                date_part = None

            # Build query
            if granularity == 'month':
                query = session.query(
                    year_part.label('year'),
                    month_part.label('month'),
                    func.count(ClosedPosition.id).label('trade_count'),
                    func.avg(ClosedPosition.pnl_multiple).label('avg_multiple'),
                    func.sum(ClosedPosition.pnl_sol).label('total_pnl')
                ).filter(
                    ClosedPosition.exit_time >= start_date
                ).group_by(year_part, month_part)
            else:
                query = session.query(
                    year_part.label('year'),
                    month_part.label('month'),
                    date_part.label('day'),
                    func.count(ClosedPosition.id).label('trade_count'),
                    func.avg(ClosedPosition.pnl_multiple).label('avg_multiple'),
                    func.sum(ClosedPosition.pnl_sol).label('total_pnl')
                ).filter(
                    ClosedPosition.exit_time >= start_date
                ).group_by(year_part, month_part, date_part)

            df = pd.read_sql(query.statement, session.bind)
            return df
        finally:
            session.close()

    def get_tracker_progress(self) -> Dict[str, Any]:
        """
        Get tracker progress from JSON file

        Returns:
            Dictionary with progress data
        """
        try:
            import json
            from pathlib import Path

            progress_file = Path('data/tracking_progress.json')
            if not progress_file.exists():
                return {'completed': [], 'last_update': None}

            with open(progress_file, 'r', encoding='utf-8') as f:
                progress = json.load(f)

            return progress
        except Exception:
            return {'completed': [], 'last_update': None}


# Global instance
_data_manager: Optional[DataManager] = None


def get_data_manager() -> DataManager:
    """
    Get or create global DataManager instance

    Returns:
        DataManager instance
    """
    global _data_manager
    if _data_manager is None:
        _data_manager = DataManager()
    return _data_manager
