import sys
from pathlib import Path
# Add parent directory to path for imports within core
sys.path.insert(0, str(Path(__file__).parent.parent))
"""
Wallet Performance Analyzer - Analiza wallets desconocidas y calcula sus métricas

Este módulo:
1. Obtiene trades históricos de una wallet
2. Empareja buys/sells para crear posiciones
3. Calcula métricas: win rate, 3x+ rate, PnL, etc.
4. Genera un score de performance
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from core.wallet_tracker import WalletTracker


class WalletPosition:
    """Representa una posición (buy + sell) de una wallet"""

    def __init__(self, token_address: str, buy_tx: Dict, sell_tx: Dict = None):
        self.token_address = token_address
        self.buy_tx = buy_tx
        self.sell_tx = sell_tx

        # Buy info
        self.buy_time = buy_tx.get('timestamp')
        self.buy_price = buy_tx.get('price')
        self.buy_amount_sol = buy_tx.get('amount_sol', 0)
        self.buy_amount_token = buy_tx.get('amount_token', 0)

        # Sell info (si existe)
        if sell_tx:
            self.sell_time = sell_tx.get('timestamp')
            self.sell_price = sell_tx.get('price')
            self.sell_amount_sol = sell_tx.get('amount_sol', 0)
            self.sell_amount_token = sell_tx.get('amount_token', 0)
            self.is_closed = True
        else:
            self.sell_time = None
            self.sell_price = None
            self.sell_amount_sol = 0
            self.sell_amount_token = 0
            self.is_closed = False

    @property
    def hold_time_seconds(self) -> Optional[float]:
        """Tiempo que ha mantenido la posición"""
        if self.is_closed and self.sell_time and self.buy_time:
            delta = self.sell_time - self.buy_time
            return delta.total_seconds()
        return None

    @property
    def hold_time_hours(self) -> Optional[float]:
        """Tiempo en horas"""
        if self.hold_time_seconds:
            return self.hold_time_seconds / 3600
        return None

    @property
    def pnl_sol(self) -> Optional[float]:
        """Profit/Loss en SOL"""
        if self.is_closed:
            return self.sell_amount_sol - self.buy_amount_sol
        return None

    @property
    def pnl_multiple(self) -> Optional[float]:
        """Múltiplo de retorno (2x = 2.0)"""
        if self.is_closed and self.buy_amount_sol > 0:
            return self.sell_amount_sol / self.buy_amount_sol
        return None

    @property
    def is_profitable(self) -> Optional[bool]:
        """Si fue rentable"""
        if self.pnl_multiple:
            return self.pnl_multiple > 1.0
        return None

    @property
    def is_diamond_hand(self) -> Optional[bool]:
        """Si es Diamond Hand (3x+ y >5min hold)"""
        if self.pnl_multiple and self.hold_time_seconds:
            return self.pnl_multiple >= 3.0 and self.hold_time_seconds > 300
        return None


class WalletAnalyzer:
    """Analiza el performance de una wallet"""

    def __init__(self):
        self.tracker = None

    async def get_wallet_trades(
        self,
        wallet_address: str,
        days_back: int = 30,
        max_signatures: int = 1000
    ) -> List[Dict]:
        """
        Obtiene todas las trades de una wallet

        Args:
            wallet_address: Wallet a analizar
            days_back: Días hacia atrás
            max_signatures: Máximo de firmas a obtener

        Returns:
            List of trade dicts
        """
        if self.tracker is None:
            self.tracker = WalletTracker()

        trades = []

        try:
            async with self.tracker:
                # Get signatures
                sigs = await self.tracker.get_signatures_for_address(
                    wallet_address,
                    limit=max_signatures
                )

                if not sigs:
                    return trades

                print(f"[*] Obteniendo {len(sigs)} transacciones para {wallet_address[:8]}...")

                # Fetch transactions in batches
                batch_size = 50

                for i in range(0, len(sigs), batch_size):
                    batch = sigs[i:i+batch_size]

                    for sig_data in batch:
                        sig = sig_data.get('signature')
                        if not sig:
                            continue

                        try:
                            tx = await self.tracker.get_transaction(sig)
                            if not tx:
                                continue

                            # Parse trade
                            trade = self.tracker.parser.parse_transaction(tx, wallet_address)
                            if trade:
                                trades.append(trade)

                        except Exception as e:
                            continue

                    # Small delay to avoid rate limits
                    if i + batch_size < len(sigs):
                        await asyncio.sleep(0.5)

        except Exception as e:
            print(f"[!] Error getting wallet trades: {e}")

        return trades

    def match_positions(self, trades: List[Dict]) -> Tuple[List[WalletPosition], List[WalletPosition]]:
        """
        Empareja buys y sells para crear posiciones

        Args:
            trades: List of trades

        Returns:
            Tuple[closed_positions, open_positions]
        """
        # Group by token
        token_trades = defaultdict(list)

        for trade in trades:
            token_trades[trade['token_address']].append(trade)

        # Sort each token's trades by time
        closed_positions = []
        open_positions = []

        for token_addr, token_trades_list in token_trades.items():
            # Sort by timestamp
            sorted_trades = sorted(
                token_trades_list,
                key=lambda x: x.get('timestamp', datetime.min)
            )

            # Match buys with sells (FIFO)
            buy_queue = []

            for trade in sorted_trades:
                if trade['operation'] == 'buy':
                    buy_queue.append(trade)
                elif trade['operation'] == 'sell' and buy_queue:
                    # Match with oldest buy
                    buy = buy_queue.pop(0)
                    pos = WalletPosition(token_addr, buy, trade)
                    closed_positions.append(pos)

            # Remaining buys are open positions
            for buy in buy_queue:
                pos = WalletPosition(token_addr, buy)
                open_positions.append(pos)

        return closed_positions, open_positions

    def calculate_metrics(self, closed_positions: List[WalletPosition]) -> Dict:
        """
        Calcula métricas de performance

        Args:
            closed_positions: List of closed positions

        Returns:
            Dict with metrics
        """
        if not closed_positions:
            return {
                'total_trades': 0,
                'total_volume_sol': 0,
                'win_rate': 0,
                'three_x_rate': 0,
                'avg_hold_time_hours': 0,
                'total_pnl_sol': 0,
                'total_pnl_multiple': 0,
                'diamond_hands_count': 0,
                'best_trade_multiple': 0,
                'worst_trade_multiple': 0
            }

        # Basic stats
        total_trades = len(closed_positions)
        total_volume_sol = sum(p.buy_amount_sol for p in closed_positions)

        # Win rate
        profitable_trades = sum(1 for p in closed_positions if p.is_profitable)
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0

        # 3x+ rate
        three_x_trades = sum(1 for p in closed_positions if p.pnl_multiple and p.pnl_multiple >= 3.0)
        three_x_rate = three_x_trades / total_trades if total_trades > 0 else 0

        # Hold time
        hold_times = [p.hold_time_hours for p in closed_positions if p.hold_time_hours]
        avg_hold_time_hours = sum(hold_times) / len(hold_times) if hold_times else 0

        # PnL
        total_pnl_sol = sum(p.pnl_sol for p in closed_positions if p.pnl_sol)
        total_pnl_multiple = sum(p.pnl_multiple for p in closed_positions if p.pnl_multiple) / total_trades if total_trades > 0 else 0

        # Diamond Hands
        diamond_hands_count = sum(1 for p in closed_positions if p.is_diamond_hand)

        # Best/worst trades
        multiples = [p.pnl_multiple for p in closed_positions if p.pnl_multiple]
        best_trade_multiple = max(multiples) if multiples else 0
        worst_trade_multiple = min(multiples) if multiples else 0

        return {
            'total_trades': total_trades,
            'total_volume_sol': total_volume_sol,
            'win_rate': win_rate,
            'three_x_rate': three_x_rate,
            'avg_hold_time_hours': avg_hold_time_hours,
            'total_pnl_sol': total_pnl_sol,
            'total_pnl_multiple': total_pnl_multiple,
            'diamond_hands_count': diamond_hands_count,
            'best_trade_multiple': best_trade_multiple,
            'worst_trade_multiple': worst_trade_multiple
        }

    def calculate_performance_score(self, metrics: Dict) -> float:
        """
        Calcula un score de performance 0-100

        Args:
            metrics: Dict con métricas

        Returns:
            Score 0-100
        """
        score = 0

        # Win rate (30%)
        score += metrics['win_rate'] * 30

        # 3x+ rate (30%)
        score += metrics['three_x_rate'] * 30

        # Total trades (15%) - more trades = more reliable
        trade_score = min(metrics['total_trades'] / 100, 1.0) * 15
        score += trade_score

        # Volume (10%)
        volume_score = min(metrics['total_volume_sol'] / 1000, 1.0) * 10
        score += volume_score

        # Average multiple (15%)
        mult_score = min(metrics['total_pnl_multiple'] / 2, 1.0) * 15
        score += mult_score

        return round(score, 2)

    async def analyze_wallet(
        self,
        wallet_address: str,
        days_back: int = 30
    ) -> Optional[Dict]:
        """
        Analiza completamente una wallet

        Args:
            wallet_address: Wallet a analizar
            days_back: Días de historial

        Returns:
            Dict con análisis completo o None si error
        """
        print(f"\n[*] Analizando wallet {wallet_address[:8]}...{wallet_address[-8:]}")

        # Get trades
        trades = await self.get_wallet_trades(wallet_address, days_back)

        if not trades:
            print(f"    [!] No trades found")
            return None

        print(f"    [+] {len(trades)} trades encontradas")

        # Match positions
        closed_positions, open_positions = self.match_positions(trades)

        print(f"    [+] {len(closed_positions)} posiciones cerradas")
        print(f"    [+] {len(open_positions)} posiciones abiertas")

        # Calculate metrics
        metrics = self.calculate_metrics(closed_positions)

        print(f"    [+] Win Rate: {metrics['win_rate']:.1%}")
        print(f"    [+] 3x+ Rate: {metrics['three_x_rate']:.1%}")
        print(f"    [+] Total PnL: {metrics['total_pnl_sol']:.2f} SOL")

        # Calculate score
        score = self.calculate_performance_score(metrics)

        return {
            'wallet_address': wallet_address,
            'metrics': metrics,
            'performance_score': score,
            'closed_positions': len(closed_positions),
            'open_positions': len(open_positions),
            'analyzed_at': datetime.now()
        }


async def main():
    """Test wallet analyzer"""
    import sys
    import io

    # Fix Windows encoding
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("=" * 70)
    print("WALLET ANALYZER - TEST")
    print("=" * 70)

    analyzer = WalletAnalyzer()

    # Test with a known KOL wallet
    test_wallet = "Dgehc8YMv6dHsiPJVoumvq4pSBkMVvrTgTUg7wdcYJPJ"  # omar

    result = await analyzer.analyze_wallet(test_wallet, days_back=30)

    if result:
        print("\n" + "=" * 70)
        print("RESULTADO DEL ANÁLISIS")
        print("=" * 70)

        m = result['metrics']

        print(f"\nWallet: {result['wallet_address']}")
        print(f"Performance Score: {result['performance_score']:.1f}/100")

        print(f"\nMétricas:")
        print(f"  Total trades: {m['total_trades']}")
        print(f"  Volumen total: {m['total_volume_sol']:.2f} SOL")
        print(f"  Win rate: {m['win_rate']:.1%}")
        print(f"  3x+ rate: {m['three_x_rate']:.1%}")
        print(f"  Avg hold time: {m['avg_hold_time_hours']:.1f}h")
        print(f"  Total PnL: {m['total_pnl_sol']:.2f} SOL")
        print(f"  Avg multiple: {m['total_pnl_multiple']:.2f}x")
        print(f"  Diamond Hands: {m['diamond_hands_count']}")
        print(f"  Best trade: {m['best_trade_multiple']:.2f}x")
        print(f"  Worst trade: {m['worst_trade_multiple']:.2f}x")


if __name__ == "__main__":
    asyncio.run(main())
