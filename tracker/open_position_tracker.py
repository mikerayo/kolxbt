"""
Open Position Tracker - Real-time monitoring of active positions

Tracks:
- When KOLs buy tokens (open positions)
- Current price movements
- Peak prices reached
- Converts to closed position when sold
- Analyzes trade quality
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import aiohttp

from database import db, Trade, OpenPosition, ClosedPosition, KOL
from config import RPC_URL


class OpenPositionTracker:
    """
    Tracks and monitors open positions in real-time
    """

    def __init__(self):
        self.session = db.get_session()
        self.rpc_url = RPC_URL

    async def process_new_trade(self, trade: Trade):
        """
        Process a new trade and update open positions

        Args:
            trade: New trade object from database
        """
        if trade.operation == 'buy':
            await self._handle_buy(trade)
        elif trade.operation == 'sell':
            await self._handle_sell(trade)

    async def _handle_buy(self, trade: Trade):
        """
        Handle a buy trade - create or update open position
        """
        # Check if there's already an open position for this KOL+token
        existing = self.session.query(OpenPosition).filter(
            OpenPosition.kol_id == trade.kol_id,
            OpenPosition.token_address == trade.token_address
        ).first()

        if existing:
            # Update existing position (averaging in)
            existing.entry_amount_sol += trade.amount_sol
            existing.entry_amount_token += trade.amount_token

            # Recalculate average entry price
            if existing.entry_amount_token > 0:
                existing.entry_price = existing.entry_amount_sol / existing.entry_amount_token

            existing.last_price_update = datetime.utcnow()
            self.session.commit()
            print(f"[+] Updated open position: {existing}")
        else:
            # Create new open position
            position = OpenPosition(
                kol_id=trade.kol_id,
                token_address=trade.token_address,
                entry_time=trade.timestamp,
                entry_price=trade.price,
                entry_amount_sol=trade.amount_sol,
                entry_amount_token=trade.amount_token,
                current_price=trade.price,
                peak_price=trade.price,
                peak_price_time=trade.timestamp,
                last_price_update=datetime.utcnow(),
                dex=trade.dex
            )
            self.session.add(position)
            self.session.commit()
            print(f"[+] New open position created: {position}")

    async def _handle_sell(self, trade: Trade):
        """
        Handle a sell trade - close position and analyze quality
        """
        # Find corresponding open position
        open_pos = self.session.query(OpenPosition).filter(
            OpenPosition.kol_id == trade.kol_id,
            OpenPosition.token_address == trade.token_address
        ).first()

        if not open_pos:
            print(f"[!] No open position found for sell: KOL {trade.kol_id}, token {trade.token_address[:8]}...")
            return

        # Calculate metrics
        hold_time_seconds = (trade.timestamp - open_pos.entry_time).total_seconds()
        pnl_sol = trade.amount_sol - open_pos.entry_amount_sol
        pnl_multiple = trade.amount_sol / open_pos.entry_amount_sol if open_pos.entry_amount_sol > 0 else 0

        # Create closed position
        closed = ClosedPosition(
            kol_id=trade.kol_id,
            token_address=trade.token_address,
            entry_time=open_pos.entry_time,
            entry_price=open_pos.entry_price,
            exit_time=trade.timestamp,
            exit_price=trade.price,
            hold_time_seconds=hold_time_seconds,
            pnl_sol=pnl_sol,
            pnl_multiple=pnl_multiple,
            dex=trade.dex or open_pos.dex
        )
        self.session.add(closed)

        # Analyze trade quality
        await self._analyze_trade_quality(open_pos, closed, trade)

        # Delete open position
        self.session.delete(open_pos)
        self.session.commit()

        print(f"[+] Closed position: {closed}")

    async def _analyze_trade_quality(self, open_pos: OpenPosition, closed: ClosedPosition, sell_trade: Trade):
        """
        Analyze the quality of the trade

        Determines:
        - Did they sell too early? (price went up after)
        - Did they sell at peak? (perfect timing)
        - Did they hold too long? (gave back profits)
        """
        # TODO: In a full implementation, you would:
        # 1. Query price history for the token after they sold
        # 2. Find the peak price in the next 24-48 hours
        # 3. Calculate how much profit they missed

        # For now, use the peak we tracked while position was open
        peak_after_exit = open_pos.peak_price

        if peak_after_exit and closed.exit_price > 0:
            peak_multiple = peak_after_exit / closed.exit_price

            # If price went up >10% after they sold, they sold early
            sold_early = 1 if peak_multiple > 1.10 else 0

            # If they sold within 5% of peak, it's perfect timing
            sold_at_peak = 1 if peak_multiple <= 1.05 else 0

            # Calculate missed profit
            if sold_early:
                tokens_owned = open_pos.entry_amount_token
                missed_profit_sol = tokens_owned * (peak_after_exit - closed.exit_price)
                missed_profit_pct = ((peak_after_exit / closed.exit_price) - 1) * 100
            else:
                missed_profit_sol = 0
                missed_profit_pct = 0

            # Calculate quality score (0-100)
            # Base score on profit multiple
            base_score = min(100, closed.pnl_multiple * 20)

            # Adjust based on timing
            if sold_at_peak:
                timing_bonus = 20
            elif sold_early:
                timing_penalty = min(30, missed_profit_pct / 2)
                timing_bonus = -timing_penalty
            else:
                timing_bonus = 0

            quality_score = max(0, min(100, base_score + timing_bonus))

            # Create quality record
            quality = TradeQuality(
                kol_id=closed.kol_id,
                closed_position_id=closed.id,
                token_address=closed.token_address,
                entry_time=closed.entry_time,
                entry_price=closed.entry_price,
                exit_time=closed.exit_time,
                exit_price=closed.exit_price,
                peak_price_after_exit=peak_after_exit,
                sold_at_peak=sold_at_peak,
                sold_early=sold_early,
                held_too_long=0,  # TODO: implement
                missed_profit_sol=missed_profit_sol,
                missed_profit_percentage=missed_profit_pct,
                quality_score=quality_score,
                timing_score=max(0, min(100, 50 + timing_bonus))
            )
            self.session.add(quality)

    async def update_current_prices(self):
        """
        Update current prices for all open positions

        This should be called periodically (e.g., every 1-5 minutes)
        """
        open_positions = self.session.query(OpenPosition).all()

        if not open_positions:
            return

        print(f"[*] Updating prices for {len(open_positions)} open positions...")

        for pos in open_positions:
            try:
                # Get current price from DEX
                current_price = await self._get_token_price(pos.token_address)

                if current_price:
                    pos.current_price = current_price
                    pos.last_price_update = datetime.utcnow()

                    # Update peak if current is higher
                    if pos.peak_price is None or current_price > pos.peak_price:
                        pos.peak_price = current_price
                        pos.peak_price_time = datetime.utcnow()

                    self.session.commit()

            except Exception as e:
                print(f"[!] Error updating price for {pos.token_address[:8]}...: {e}")

        self.session.commit()

    async def _get_token_price(self, token_address: str) -> Optional[float]:
        """
        Get current price of token from Raydium/Jupiter

        TODO: Implement actual price fetching
        For now, returns None (will use last known price)
        """
        # In a full implementation, you would:
        # 1. Query Raydium API for liquidity pool price
        # 2. Or query Jupiter API for quote
        # 3. Or use a price aggregator

        return None

    def get_open_positions_summary(self) -> List[Dict[str, Any]]:
        """
        Get summary of all open positions

        Returns:
            List of dicts with position info
        """
        positions = self.session.query(OpenPosition).all()

        summary = []
        for pos in positions:
            kol = self.session.query(KOL).get(pos.kol_id)

            summary.append({
                'kol_name': kol.name if kol else 'Unknown',
                'token_address': pos.token_address[:8] + '...',
                'entry_time': pos.entry_time,
                'entry_amount_sol': pos.entry_amount_sol,
                'current_value_sol': pos.current_value_sol,
                'unrealized_pnl_sol': pos.unrealized_pnl_sol,
                'unrealized_pnl_multiple': pos.unrealized_pnl_multiple,
                'peak_multiple_reached': pos.peak_multiple_reached,
                'hold_time_hours': pos.hold_time_hours,
                'dex': pos.dex
            })

        return summary

    def close(self):
        """Close database session"""
        self.session.close()


async def update_prices_periodically(interval_seconds: int = 300):
    """
    Background task to update prices periodically

    Args:
        interval_seconds: How often to update (default: 5 minutes)
    """
    tracker = OpenPositionTracker()

    while True:
        try:
            await tracker.update_current_prices()
            await asyncio.sleep(interval_seconds)
        except Exception as e:
            print(f"[!] Error in price update loop: {e}")
            await asyncio.sleep(interval_seconds)


if __name__ == "__main__":
    # Test: update prices every 5 minutes
    asyncio.run(update_prices_periodically(300))
