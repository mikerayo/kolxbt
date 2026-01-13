import sys
from pathlib import Path
# Add parent directory to path for imports within core
sys.path.insert(0, str(Path(__file__).parent.parent))
"""
Feature Engineering for KOL Trader Analysis
Calculates metrics like hold time, PnL, win rate, 3x+ rate, etc.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from sqlalchemy.orm import Session

from core.database import KOL, Trade, ClosedPosition, db
from core.config import (
    MIN_HOLD_TIME_SECONDS, TARGET_MULTIPLE,
    DIAMOND_HAND_MIN_HOLD_TIME, DIAMOND_HAND_MIN_3X_RATE,
    SCALPER_HOLD_TIME_THRESHOLD
)


class KOLFeatures:
    """
    Feature calculator for individual KOL analysis
    """

    def __init__(self, kol: KOL, session: Session):
        """
        Initialize feature calculator for a KOL

        Args:
            kol: KOL object
            session: Database session
        """
        self.kol = kol
        self.session = session
        self._trades = None
        self._closed_positions = None

    @property
    def trades(self) -> List[Trade]:
        """Get all trades for this KOL"""
        if self._trades is None:
            self._trades = self.session.query(Trade).filter(
                Trade.kol_id == self.kol.id
            ).order_by(Trade.timestamp).all()
        return self._trades

    @property
    def closed_positions(self) -> List[ClosedPosition]:
        """Get all closed positions for this KOL"""
        if self._closed_positions is None:
            self._closed_positions = self.session.query(ClosedPosition).filter(
                ClosedPosition.kol_id == self.kol.id
            ).all()
        return self._closed_positions

    def calculate_all_features(self) -> Dict[str, float]:
        """
        Calculate all features for a KOL

        Returns:
            Dictionary with all calculated features
        """
        features = {
            # Basic info
            'kol_id': self.kol.id,
            'name': self.kol.name,
            'wallet_address': self.kol.wallet_address,
            'total_trades': len(self.closed_positions),
        }

        if not self.closed_positions:
            # No closed positions yet
            features.update({
                'avg_hold_time_hours': 0,
                'median_hold_time_hours': 0,
                'hold_time_std_hours': 0,
                'total_pnl_sol': 0,
                'win_rate': 0,
                'avg_multiple': 0,
                'three_x_plus_count': 0,
                'three_x_plus_rate': 0,
                'consistency_score': 0,
                'diamond_hand_score': 0,
                'is_diamond_hand': False,
                'is_scalper': False,
            })
            return features

        # Calculate hold time metrics
        hold_times = [pos.hold_time_hours for pos in self.closed_positions]
        features['avg_hold_time_hours'] = float(np.mean(hold_times))
        features['median_hold_time_hours'] = float(np.median(hold_times))
        features['hold_time_std_hours'] = float(np.std(hold_times))

        # Calculate PnL metrics
        pnls = [pos.pnl_multiple for pos in self.closed_positions]
        total_pnl = sum(pos.pnl_sol for pos in self.closed_positions)
        winning_trades = sum(1 for pos in self.closed_positions if pos.is_profitable)

        features['total_pnl_sol'] = float(total_pnl)
        features['win_rate'] = winning_trades / len(self.closed_positions)
        features['avg_multiple'] = float(np.mean(pnls))

        # Calculate 3x+ metrics
        three_x_trades = [pos for pos in self.closed_positions if pos.pnl_multiple >= TARGET_MULTIPLE]
        features['three_x_plus_count'] = len(three_x_trades)
        features['three_x_plus_rate'] = len(three_x_trades) / len(self.closed_positions)

        if three_x_trades:
            three_x_hold_times = [pos.hold_time_hours for pos in three_x_trades]
            features['three_x_avg_hold_hours'] = float(np.mean(three_x_hold_times))
        else:
            features['three_x_avg_hold_hours'] = 0.0

        # Calculate consistency score (inverse of normalized std dev)
        if len(pnls) > 1:
            pnl_std = np.std(pnls)
            pnl_mean = np.mean(pnls)
            if pnl_mean != 0:
                features['consistency_score'] = float(1 / (1 + abs(pnl_std / pnl_mean)))
            else:
                features['consistency_score'] = 0.0
        else:
            features['consistency_score'] = 0.0

        # Calculate classification
        features['is_diamond_hand'] = (
            features['avg_hold_time_hours'] >= (DIAMOND_HAND_MIN_HOLD_TIME / 3600) and
            features['three_x_plus_rate'] >= DIAMOND_HAND_MIN_3X_RATE
        )

        # Check if scalper (high frequency, short holds)
        if len(self.closed_positions) >= 10:  # Need minimum sample size
            avg_hold_seconds = features['avg_hold_time_hours'] * 3600
            features['is_scalper'] = avg_hold_seconds < SCALPER_HOLD_TIME_THRESHOLD
        else:
            features['is_scalper'] = False

        return features

    def get_trade_summary(self) -> pd.DataFrame:
        """
        Get a summary DataFrame of all trades

        Returns:
            DataFrame with trade information
        """
        if not self.closed_positions:
            return pd.DataFrame()

        data = []
        for pos in self.closed_positions:
            data.append({
                'token_address': pos.token_address,
                'entry_time': pos.entry_time,
                'exit_time': pos.exit_time,
                'hold_time_hours': pos.hold_time_hours,
                'entry_price': pos.entry_price,
                'exit_price': pos.exit_price,
                'pnl_sol': pos.pnl_sol,
                'pnl_multiple': pos.pnl_multiple,
                'is_diamond_hand': pos.is_diamond_hand,
            })

        return pd.DataFrame(data)


class PositionMatcher:
    """
    Matches buy and sell trades to create closed positions
    """

    @staticmethod
    def match_positions(kol: KOL, session: Session) -> List[ClosedPosition]:
        """
        Match buy and sell trades to create closed positions

        Uses FIFO (First In, First Out) matching strategy

        Args:
            kol: KOL object
            session: Database session

        Returns:
            List of ClosedPosition objects
        """
        # Get all trades sorted by timestamp
        trades = session.query(Trade).filter(
            Trade.kol_id == kol.id
        ).order_by(Trade.timestamp).all()

        # Group by token
        token_trades: Dict[str, List[Trade]] = {}
        for trade in trades:
            if trade.token_address not in token_trades:
                token_trades[trade.token_address] = []
            token_trades[trade.token_address].append(trade)

        # Match positions for each token
        closed_positions = []

        for token_address, trades_list in token_trades.items():
            positions = PositionMatcher._match_token_positions(
                kol, token_address, trades_list
            )
            closed_positions.extend(positions)

        # Save to database
        for pos in closed_positions:
            # Check if position already exists
            existing = session.query(ClosedPosition).filter(
                ClosedPosition.kol_id == kol.id,
                ClosedPosition.token_address == pos.token_address,
                ClosedPosition.entry_time == pos.entry_time,
                ClosedPosition.exit_time == pos.exit_time
            ).first()

            if not existing:
                session.add(pos)

        session.commit()

        return closed_positions

    @staticmethod
    def _match_token_positions(
        kol: KOL,
        token_address: str,
        trades: List[Trade]
    ) -> List[ClosedPosition]:
        """
        Match positions for a single token using FIFO

        Args:
            kol: KOL object
            token_address: Token mint address
            trades: List of trades for this token

        Returns:
            List of ClosedPosition objects
        """
        buy_queue = []  # Queue of unmatched buys
        positions = []

        for trade in trades:
            if trade.operation == 'buy':
                # Add to buy queue
                buy_queue.append(trade)

            elif trade.operation == 'sell' and buy_queue:
                # Match with oldest buy (FIFO)
                buy_trade = buy_queue.pop(0)

                # Calculate position metrics
                hold_time = (trade.timestamp - buy_trade.timestamp).total_seconds()
                entry_price = buy_trade.price or 0
                exit_price = trade.price or 0

                # Calculate PnL
                # PnL = (exit_amount - entry_amount_in_sol)
                # For a buy: we spent SOL to get tokens
                # For a sell: we got SOL for tokens
                amount_spent = buy_trade.amount_sol
                amount_received = trade.amount_sol
                pnl_sol = amount_received - amount_spent

                # Calculate multiple
                if entry_price > 0 and exit_price > 0:
                    pnl_multiple = exit_price / entry_price
                elif amount_spent > 0:
                    pnl_multiple = amount_received / amount_spent
                else:
                    pnl_multiple = 1.0

                # Create closed position
                position = ClosedPosition(
                    kol_id=kol.id,
                    token_address=token_address,
                    entry_time=buy_trade.timestamp,
                    exit_time=trade.timestamp,
                    hold_time_seconds=hold_time,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    pnl_sol=pnl_sol,
                    pnl_multiple=pnl_multiple,
                    dex=trade.dex or buy_trade.dex
                )

                positions.append(position)

        return positions


def calculate_features_for_all_kols(session: Session = None) -> pd.DataFrame:
    """
    Calculate features for all KOLs in database

    Args:
        session: Database session (creates new if None)

    Returns:
        DataFrame with all KOL features
    """
    if session is None:
        session = db.get_session()
        should_close = True
    else:
        should_close = False

    try:
        kols = db.get_all_kols(session)
        features_list = []

        for kol in kols:
            calculator = KOLFeatures(kol, session)
            features = calculator.calculate_all_features()
            features_list.append(features)

        df = pd.DataFrame(features_list)
        return df
    finally:
        if should_close:
            session.close()


def match_positions_for_all_kols(session: Session = None) -> int:
    """
    Match positions for all KOLs

    Args:
        session: Database session (creates new if None)

    Returns:
        Number of KOLs processed
    """
    if session is None:
        session = db.get_session()
        should_close = True
    else:
        should_close = False

    try:
        kols = db.get_all_kols(session)
        processed = 0

        for kol in kols:
            try:
                PositionMatcher.match_positions(kol, session)
                processed += 1
                print(f"[*] Matched positions for {kol.name}")
            except Exception as e:
                print(f"[!] Error matching positions for {kol.name}: {e}")

        return processed
    finally:
        if should_close:
            session.close()


if __name__ == "__main__":
    # Test feature engineering
    print("=" * 70)
    print("Feature Engineering Test")
    print("=" * 70)

    # Initialize database
    db.create_tables()

    # Calculate features for all KOLs
    print("\n[*] Calculating features for all KOLs...")
    df = calculate_features_for_all_kols()

    if not df.empty:
        print(f"\n[*] Features calculated for {len(df)} KOLs")

        # Show top 10 by 3x+ rate
        top_3x = df.nlargest(10, 'three_x_plus_rate')
        print("\n🏆 Top 10 KOLs by 3x+ Rate:")
        print("-" * 70)
        for _, row in top_3x.iterrows():
            print(f"{row['name']}: {row['three_x_plus_rate']:.1%} ({row['three_x_plus_count']} trades)")

        # Show diamond hands
        diamond_hands = df[df['is_diamond_hand']]
        print(f"\n💎 Diamond Hands ({len(diamond_hands)} KOLs):")
        print("-" * 70)
        for _, row in diamond_hands.head(10).iterrows():
            print(f"{row['name']}: {row['three_x_plus_rate']:.1%} 3x+, {row['avg_hold_time_hours']:.1f}h avg hold")
    else:
        print("\n[-] No KOLs with closed positions found")
        print("[!] You need to run wallet_tracker first to collect trade data")
