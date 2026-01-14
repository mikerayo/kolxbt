"""
Backtesting Engine for KOL Tracker ML

Simula y evalúa estrategias de trading basadas en KOLs:
- Seguir top KOLs (comprar cuando compran, vender cuando venden)
- Buy & hold strategies
- Comparación vs benchmarks (SOL, BTC)

Calcula métricas financieras avanzadas:
- ROI, CAGR, Sharpe Ratio, Sortino Ratio
- Maximum Drawdown, Win Rate, Profit Factor
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from sqlalchemy import func, and_

from core.database import db, KOL, Trade, ClosedPosition
import logging

logger = logging.getLogger(__name__)


@dataclass
class BacktestResults:
    """Resultados de backtesting de una estrategia"""
    strategy_name: str
    start_date: datetime
    end_date: datetime

    # Metrics básicas
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    # Returns
    total_return: float  # %
    avg_return_per_trade: float  # %
    median_return: float  # %

    # Risk metrics
    volatility: float  # annualizada
    max_drawdown: float  # %
    avg_drawdown: float  # %

    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Trade metrics
    best_trade: float  # %
    worst_trade: float  # %
    profit_factor: float
    expectancy: float

    # Advanced
    cagr: float
    mar_ratio: float

    # Datos crudos
    equity_curve: pd.Series
    returns_series: pd.Series
    trades_df: pd.DataFrame


class StrategyBacktester:
    """
    Simula estrategias de trading basadas en KOLs
    """

    def __init__(self):
        self.session = db.get_session()

    def backtest_follow_kols(
        self,
        top_n: int = 10,
        min_score: float = 70.0,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        initial_capital: float = 1000.0
    ) -> BacktestResults:
        """
        Backtestea estrategia: "Comprar cuando top KOLs compran, vender cuando venden"

        Args:
            top_n: Número de KOLs top a seguir
            min_score: Score mínimo para considerar un KOL
            start_date: Fecha de inicio del backtest
            end_date: Fecha de fin del backtest
            initial_capital: Capital inicial en SOL

        Returns:
            BacktestResults con todas las métricas
        """
        logger.info(f"Backtesting 'follow top {top_n} KOLs' strategy...")

        # Obtener top KOLs
        top_kols = self._get_top_kols(top_n, min_score)

        if not top_kols:
            logger.warning("No KOLs found matching criteria")
            return self._empty_results("follow_kols", start_date, end_date)

        # Obtener trades de estos KOLs
        trades = self._get_kols_trades(
            [k['id'] for k in top_kols],
            start_date,
            end_date
        )

        if not trades:
            logger.warning("No trades found for backtesting period")
            return self._empty_results("follow_kols", start_date, end_date)

        # Simular estrategia
        equity_curve, returns_series, trades_df = self._simulate_follow_strategy(
            trades,
            initial_capital
        )

        # Calcular métricas
        metrics = self._calculate_metrics(
            equity_curve,
            returns_series,
            trades_df
        )

        return BacktestResults(
            strategy_name="follow_top_kols",
            start_date=start_date or trades[0].timestamp,
            end_date=end_date or trades[-1].timestamp,
            **metrics,
            equity_curve=equity_curve,
            returns_series=returns_series,
            trades_df=trades_df
        )

    def backtest_buy_and_hold(
        self,
        hold_periods: List[timedelta] = None,
        top_n: int = 10,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        initial_capital: float = 1000.0
    ) -> Dict[timedelta, BacktestResults]:
        """
        Backtestea estrategia: "Buy & Hold" por diferentes períodos

        Args:
            hold_periods: Lista de períodos de hold (default: 1h, 24h, 7d, 30d)
            top_n: Número de KOLs top para señales de entrada
            start_date: Fecha de inicio
            end_date: Fecha de fin
            initial_capital: Capital inicial

        Returns:
            Dict con períodos como keys y BacktestResults como values
        """
        if hold_periods is None:
            hold_periods = [
                timedelta(hours=1),
                timedelta(hours=24),
                timedelta(days=7),
                timedelta(days=30)
            ]

        logger.info(f"Backtesting 'buy & hold' for {len(hold_periods)} periods...")

        results = {}

        for hold_period in hold_periods:
            logger.info(f"Testing hold period: {hold_period}")

            # Obtener top KOLs
            top_kols = self._get_top_kols(top_n, min_score=0.0)

            if not top_kols:
                continue

            # Obtener buys de estos KOLs
            buys = self._get_kols_buys(
                [k['id'] for k in top_kols],
                start_date,
                end_date
            )

            if not buys:
                continue

            # Simular buy & hold
            equity_curve, returns_series, trades_df = self._simulate_buy_hold(
                buys,
                hold_period,
                initial_capital
            )

            # Calcular métricas
            metrics = self._calculate_metrics(
                equity_curve,
                returns_series,
                trades_df
            )

            results[hold_period] = BacktestResults(
                strategy_name=f"buy_hold_{hold_period.total_seconds()}s",
                start_date=start_date or buys[0].timestamp,
                end_date=end_date or buys[-1].timestamp,
                **metrics,
                equity_curve=equity_curve,
                returns_series=returns_series,
                trades_df=trades_df
            )

        return results

    def compare_strategies(
        self,
        results_dict: Dict[str, BacktestResults]
    ) -> pd.DataFrame:
        """
        Compara múltiples estrategias lado a lado

        Args:
            results_dict: Dict con nombre de estrategia -> BacktestResults

        Returns:
            DataFrame con comparación
        """
        comparison_data = []

        for strategy_name, results in results_dict.items():
            comparison_data.append({
                'Strategy': strategy_name,
                'Total Return (%)': f"{results.total_return:.2f}",
                'CAGR (%)': f"{results.cagr:.2f}",
                'Sharpe Ratio': f"{results.sharpe_ratio:.2f}",
                'Sortino Ratio': f"{results.sortino_ratio:.2f}",
                'Max Drawdown (%)': f"{results.max_drawdown:.2f}",
                'Win Rate (%)': f"{results.win_rate * 100:.2f}",
                'Profit Factor': f"{results.profit_factor:.2f}",
                'Total Trades': results.total_trades
            })

        return pd.DataFrame(comparison_data)

    def _get_top_kols(self, top_n: int, min_score: float) -> List[Dict]:
        """Obtiene top N KOLs por Diamond Hand Score"""
        # Leer leaderboard
        import json
        try:
            with open('data/leaderboard.json', 'r', encoding='utf-8') as f:
                data = json.load(f)

            df = pd.DataFrame(data['leaderboard'])

            # Filtrar y ordenar
            df = df[df['diamond_hand_score'] >= min_score]
            df = df.sort_values('diamond_hand_score', ascending=False)
            df = df.head(top_n)

            return df.to_dict('records')

        except Exception as e:
            logger.error(f"Error loading leaderboard: {e}")
            return []

    def _get_kols_trades(
        self,
        kol_ids: List[int],
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> List[Trade]:
        """Obtiene trades de KOLs específicos"""
        query = self.session.query(Trade).filter(
            Trade.kol_id.in_(kol_ids)
        )

        if start_date:
            query = query.filter(Trade.timestamp >= start_date)
        if end_date:
            query = query.filter(Trade.timestamp <= end_date)

        return query.order_by(Trade.timestamp).all()

    def _get_kols_buys(
        self,
        kol_ids: List[int],
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> List[Trade]:
        """Obtiene solo trades de compra de KOLs"""
        query = self.session.query(Trade).filter(
            and_(
                Trade.kol_id.in_(kol_ids),
                Trade.operation == 'buy'
            )
        )

        if start_date:
            query = query.filter(Trade.timestamp >= start_date)
        if end_date:
            query = query.filter(Trade.timestamp <= end_date)

        return query.order_by(Trade.timestamp).all()

    def _simulate_follow_strategy(
        self,
        trades: List[Trade],
        initial_capital: float
    ) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
        """
        Simula estrategia de seguir KOLs

        Lógica:
        - Cuando KOL compra → comprar
        - Cuando mismo KOL vende → vender
        """
        capital = initial_capital
        equity_values = [initial_capital]
        returns = []
        trade_results = []

        # Agrupar trades por KOL y token
        from collections import defaultdict
        buy_trades = defaultdict(list)  # (kol_id, token_address) -> [buy_trades]

        for trade in trades:
            if trade.operation == 'buy':
                key = (trade.kol_id, trade.token_address)
                buy_trades[key].append(trade)
            elif trade.operation == 'sell':
                key = (trade.kol_id, trade.token_address)
                # Encontrar buy más antiguo sin emparejar
                if buy_trades[key]:
                    buy = buy_trades[key].pop(0)

                    # Calcular P&L de este trade
                    buy_cost = buy.amount_sol
                    sell_revenue = trade.amount_sol
                    pnl = sell_revenue - buy_cost
                    pnl_pct = (pnl / buy_cost) * 100

                    # Actualizar capital
                    capital = capital * (1 + pnl_pct / 100)
                    equity_values.append(capital)
                    returns.append(pnl_pct)

                    trade_results.append({
                        'kol_id': trade.kol_id,
                        'token_address': trade.token_address,
                        'entry_time': buy.timestamp,
                        'exit_time': trade.timestamp,
                        'hold_time_hours': (trade.timestamp - buy.timestamp).total_seconds() / 3600,
                        'pnl_sol': pnl,
                        'pnl_pct': pnl_pct,
                        'is_profitable': pnl > 0
                    })

        # Capital no invertido se mantiene
        if len(equity_values) == 1:
            equity_values.append(initial_capital)

        equity_curve = pd.Series(equity_values)
        returns_series = pd.Series(returns)
        trades_df = pd.DataFrame(trade_results)

        return equity_curve, returns_series, trades_df

    def _simulate_buy_hold(
        self,
        buys: List[Trade],
        hold_period: timedelta,
        initial_capital: float
    ) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
        """
        Simula estrategia buy & hold

        Lógica:
        - Comprar cuando KOL compra
        - Hold por período fijo
        - Vender después del período
        """
        capital = initial_capital
        equity_values = [initial_capital]
        returns = []
        trade_results = []

        for buy in buys:
            # Buscar sell más cercano después del hold period
            target_exit_time = buy.timestamp + hold_period

            sell = self.session.query(Trade).filter(
                and_(
                    Trade.kol_id == buy.kol_id,
                    Trade.token_address == buy.token_address,
                    Trade.operation == 'sell',
                    Trade.timestamp >= target_exit_time
                )
            ).order_by(Trade.timestamp).first()

            if sell:
                # Usamos el sell real
                exit_time = sell.timestamp
                exit_price = sell.price if sell.price else sell.amount_sol / sell.amount_token
            else:
                # No hay sell, estimamos usando ClosedPosition
                closed_pos = self.session.query(ClosedPosition).filter(
                    and_(
                        ClosedPosition.kol_id == buy.kol_id,
                        ClosedPosition.token_address == buy.token_address,
                        ClosedPosition.entry_time == buy.timestamp
                    )
                ).first()

                if closed_pos:
                    exit_time = closed_pos.exit_time
                    exit_price = closed_pos.exit_price
                else:
                    # No se puede calcular, skip
                    continue

            # Calcular P&L
            buy_cost = buy.amount_sol
            estimated_sell = buy_cost * (exit_price / buy.price) if buy.price else buy_cost
            pnl = estimated_sell - buy_cost
            pnl_pct = (pnl / buy_cost) * 100

            # Actualizar capital
            capital = capital * (1 + pnl_pct / 100)
            equity_values.append(capital)
            returns.append(pnl_pct)

            trade_results.append({
                'kol_id': buy.kol_id,
                'token_address': buy.token_address,
                'entry_time': buy.timestamp,
                'exit_time': exit_time,
                'hold_time_hours': hold_period.total_seconds() / 3600,
                'pnl_sol': pnl,
                'pnl_pct': pnl_pct,
                'is_profitable': pnl > 0
            })

        equity_curve = pd.Series(equity_values)
        returns_series = pd.Series(returns)
        trades_df = pd.DataFrame(trade_results)

        return equity_curve, returns_series, trades_df

    def _calculate_metrics(
        self,
        equity_curve: pd.Series,
        returns_series: pd.Series,
        trades_df: pd.DataFrame
    ) -> Dict:
        """Calcula todas las métricas financieras"""
        if len(returns_series) == 0:
            return self._empty_metrics()

        # Métricas básicas
        total_trades = len(returns_series)
        winning_trades = len(returns_series[returns_series > 0])
        losing_trades = len(returns_series[returns_series <= 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # Returns
        total_return = ((equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1) * 100
        avg_return = returns_series.mean()
        median_return = returns_series.median()

        # Volatility (annualizada asumiendo trades diarios)
        volatility = returns_series.std() * np.sqrt(365) if len(returns_series) > 1 else 0

        # Drawdowns
        max_drawdown, avg_drawdown = self._calculate_drawdowns(equity_curve)

        # Risk-adjusted returns (asumiendo risk-free rate de 2% anual)
        risk_free_rate = 0.02
        excess_return = (total_return / 100) - risk_free_rate
        sharpe_ratio = (excess_return / (volatility / 100)) if volatility > 0 else 0

        # Sortino ratio (solo downside deviation)
        downside_returns = returns_series[returns_series < 0]
        downside_deviation = downside_returns.std() * np.sqrt(365) if len(downside_returns) > 1 else 0
        sortino_ratio = (excess_return / (downside_deviation / 100)) if downside_deviation > 0 else 0

        # Calmar ratio
        calmar_ratio = (total_return / 100) / (abs(max_drawdown) / 100) if max_drawdown != 0 else 0

        # Trade metrics
        best_trade = returns_series.max() if len(returns_series) > 0 else 0
        worst_trade = returns_series.min() if len(returns_series) > 0 else 0

        gross_profit = returns_series[returns_series > 0].sum() if winning_trades > 0 else 0
        gross_loss = abs(returns_series[returns_series <= 0].sum()) if losing_trades > 0 else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        avg_win = returns_series[returns_series > 0].mean() if winning_trades > 0 else 0
        avg_loss = returns_series[returns_series <= 0].mean() if losing_trades > 0 else 0
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        # CAGR
        days = (equity_curve.index[-1] - equity_curve.index[0]).days if hasattr(equity_curve.index, 'to_pydatetime') else 365
        years = days / 365
        cagr = ((equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / years) - 1) * 100 if years > 0 else 0

        # MAR ratio
        mar_ratio = (cagr / 100) / (abs(max_drawdown) / 100) if max_drawdown != 0 else 0

        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'avg_return_per_trade': avg_return,
            'median_return': median_return,
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'best_trade': best_trade,
            'worst_trade': worst_trade,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'cagr': cagr,
            'mar_ratio': mar_ratio
        }

    def _calculate_drawdowns(self, equity_curve: pd.Series) -> Tuple[float, float]:
        """Calcula máximo y promedio de drawdowns"""
        # Calculamos drawdown en cada punto
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max * 100

        max_dd = drawdown.min()
        avg_dd = drawdown[drawdown < 0].mean() if len(drawdown[drawdown < 0]) > 0 else 0

        return max_dd, avg_dd

    def _empty_results(
        self,
        strategy_name: str,
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> BacktestResults:
        """Retorna resultados vacíos"""
        return BacktestResults(
            strategy_name=strategy_name,
            start_date=start_date or datetime.now(),
            end_date=end_date or datetime.now(),
            **self._empty_metrics(),
            equity_curve=pd.Series([1000]),
            returns_series=pd.Series(),
            trades_df=pd.DataFrame()
        )

    def _empty_metrics(self) -> Dict:
        """Retorna métricas vacías"""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_return': 0,
            'avg_return_per_trade': 0,
            'median_return': 0,
            'volatility': 0,
            'max_drawdown': 0,
            'avg_drawdown': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'calmar_ratio': 0,
            'best_trade': 0,
            'worst_trade': 0,
            'profit_factor': 0,
            'expectancy': 0,
            'cagr': 0,
            'mar_ratio': 0
        }


def run_backtest(
    strategy: str = "follow_kols",
    top_n: int = 10,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> BacktestResults:
    """
    Función helper para correr backtesting

    Args:
        strategy: "follow_kols" o "buy_hold"
        top_n: Número de KOLs top
        start_date: Fecha inicio
        end_date: Fecha fin

    Returns:
        BacktestResults
    """
    backtester = StrategyBacktester()

    if strategy == "follow_kols":
        return backtester.backtest_follow_kols(
            top_n=top_n,
            start_date=start_date,
            end_date=end_date
        )
    elif strategy == "buy_hold":
        results = backtester.backtest_buy_and_hold(
            top_n=top_n,
            start_date=start_date,
            end_date=end_date
        )
        # Retornar el mejor período
        return max(results.values(), key=lambda x: x.sharpe_ratio)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
