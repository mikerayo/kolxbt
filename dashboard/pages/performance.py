"""
Performance Analytics Dashboard

Muestra métricas de backtesting y validación del modelo:
- Model Accuracy, Precision, Recall
- Strategy Returns (Follow KOLs vs Buy & Hold)
- Risk Metrics (Sharpe, Sortino, Drawdown)
- Per-KOL Analysis
- Advanced Charts
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.backtesting import StrategyBacktester, run_backtest
from core.model_validation import ModelValidator
import json


@st.cache_data(ttl=3600)
def load_model_validation():
    """Carga validación del modelo"""
    try:
        validator = ModelValidator()
        metrics = validator.validate_predictions()
        return metrics
    except Exception as e:
        st.error(f"Error loading model validation: {e}")
        return None


@st.cache_data(ttl=1800)
def run_follow_kols_backtest(top_n: int = 10):
    """Corre backtest de estrategia Follow KOLs"""
    try:
        backtester = StrategyBacktester()
        results = backtester.backtest_follow_kols(
            top_n=top_n,
            start_date=datetime.now() - timedelta(days=90)
        )
        return results
    except Exception as e:
        st.error(f"Error running backtest: {e}")
        return None


@st.cache_data(ttl=1800)
def run_buy_hold_backtest(top_n: int = 10):
    """Corre backtest de estrategia Buy & Hold"""
    try:
        backtester = StrategyBacktester()
        results_dict = backtester.backtest_buy_and_hold(
            top_n=top_n,
            start_date=datetime.now() - timedelta(days=90)
        )
        return results_dict
    except Exception as e:
        st.error(f"Error running buy & hold backtest: {e}")
        return None


def display_model_accuracy():
    """Muestra métricas de accuracy del modelo"""
    st.markdown("### 🎯 Model Validation")

    metrics = load_model_validation()

    if not metrics:
        st.warning("⏳ Model validation not available yet. Waiting for more data...")
        return

    # Top metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        delta_color = "normal" if metrics.accuracy >= 0.7 else "inverse"
        st.metric(
            "Accuracy",
            f"{metrics.accuracy:.1%}",
            delta="Good" if metrics.accuracy >= 0.7 else "Needs Improvement",
            delta_color=delta_color
        )

    with col2:
        st.metric(
            "Precision",
            f"{metrics.precision:.1%}",
            help="Of predicted 3x+, how many actually were 3x+"
        )

    with col3:
        st.metric(
            "Recall",
            f"{metrics.recall:.1%}",
            help="Of actual 3x+, how many did we predict"
        )

    with col4:
        st.metric(
            "F1 Score",
            f"{metrics.f1_score:.2f}",
            help="Harmonic mean of precision and recall"
        )

    # Confusion Matrix
    st.markdown("#### 📊 Confusion Matrix")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Predicted vs Actual:**")

        cm_data = pd.DataFrame({
            '': ['Actual Not 3x+', 'Actual 3x+'],
            'Predicted Not 3x+': [metrics.true_negatives, metrics.false_negatives],
            'Predicted 3x+': [metrics.false_positives, metrics.true_positives]
        })

        st.dataframe(cm_data.set_index(''), use_container_width=True)

    with col2:
        # Interpretación
        total_predictions = metrics.true_positives + metrics.true_negatives + metrics.false_positives + metrics.false_negatives

        st.markdown("**Interpretation:**")
        st.info(f"""
        - **True Positives**: {metrics.true_positives} correctly predicted 3x+ trades
        - **True Negatives**: {metrics.true_negatives} correctly predicted non-3x+ trades
        - **False Positives**: {metrics.false_positives} predicted 3x+ but weren't
        - **False Negatives**: {metrics.false_negatives} missed 3x+ trades

        **Total Predictions**: {total_predictions}
        """)

    # Calibration
    st.markdown("---")
    st.markdown("#### 🎚️ Model Calibration")

    col1, col2 = st.columns(2)

    with col1:
        expected_rate = metrics.expected_positive_rate
        actual_rate = metrics.actual_positive_rate
        calib_error = metrics.calibration_error

        st.metric(
            "Expected 3x+ Rate",
            f"{expected_rate:.1%}",
            help="Average predicted probability"
        )

        st.metric(
            "Actual 3x+ Rate",
            f"{actual_rate:.1%}",
            delta=f"{calib_error:.1%} diff" if calib_error > 0 else "Well Calibrated"
        )

    with col2:
        st.markdown("**Calibration Analysis:**")

        if calib_error < 0.05:
            st.success("✅ Model is well calibrated")
        elif calib_error < 0.10:
            st.warning("⚠️ Model needs slight calibration")
        else:
            st.error("❌ Model is poorly calibrated - needs retraining")

    # Confidence breakdown
    if metrics.confidence_accuracy:
        st.markdown("---")
        st.markdown("#### 📈 Accuracy by Confidence Level")

        conf_df = pd.DataFrame([
            {'Confidence Level': level, 'Accuracy': f"{acc:.1%}"}
            for level, acc in metrics.confidence_accuracy.items()
        ])

        st.dataframe(conf_df, use_container_width=True)


def display_follow_kols_backtest():
    """Muestra resultados de backtesting Follow KOLs"""
    st.markdown("### 💰 Follow KOLs Strategy")

    # Configuración
    col1, col2 = st.columns(2)

    with col1:
        top_n = st.slider("Top N KOLs to Follow", 1, 50, 10)

    with col2:
        period_days = st.slider("Backtest Period (days)", 7, 180, 90)

    # Correr backtest
    results = run_follow_kols_backtest(top_n=top_n)

    if not results or results.total_trades == 0:
        st.warning("⏳ Not enough data for backtesting yet. Need more closed positions...")
        return

    # Top metrics
    st.markdown("#### 📊 Strategy Performance")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        delta_color = "normal" if results.total_return > 0 else "inverse"
        st.metric(
            "Total Return",
            f"{results.total_return:.1f}%",
            delta=f"{results.total_return:.1f}%" if results.total_return != 0 else None,
            delta_color=delta_color
        )

    with col2:
        st.metric(
            "Sharpe Ratio",
            f"{results.sharpe_ratio:.2f}",
            help="Risk-adjusted return (>1 is good)"
        )

    with col3:
        st.metric(
            "Max Drawdown",
            f"{results.max_drawdown:.1f}%",
            delta_color="inverse"  # lower is better
        )

    with col4:
        st.metric(
            "Win Rate",
            f"{results.win_rate:.1%}",
            help="Percentage of profitable trades"
        )

    # Equity Curve
    st.markdown("---")
    st.markdown("#### 📈 Equity Curve")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=results.equity_curve.values,
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#00CC96', width=2)
    ))

    fig.update_layout(
        title="Portfolio Value Over Time",
        xaxis_title="Trade Number",
        yaxis_title="Value (SOL)",
        hovermode='x unified',
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    # Returns Distribution
    st.markdown("---")
    st.markdown("#### 📊 Returns Distribution")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            x=results.returns_series.values,
            nbins=30,
            title="Distribution of Trade Returns (%)",
            labels={'x': 'Return (%)', 'y': 'Count'},
            color_discrete_sequence=['#4ECDC4']
        )

        # Add line at 0%
        fig.add_vline(x=0, line_dash="dash", line_color="red")

        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Return Statistics:**")

        st.metric(
            "Average Return",
            f"{results.avg_return_per_trade:.1f}%"
        )

        st.metric(
            "Median Return",
            f"{results.median_return:.1f}%"
        )

        st.metric(
            "Best Trade",
            f"{results.best_trade:.1f}%"
        )

        st.metric(
            "Worst Trade",
            f"{results.worst_trade:.1f}%"
        )

    # Trade History
    if not results.trades_df.empty:
        st.markdown("---")
        st.markdown("#### 📜 Trade History")

        # Mostrar últimos 20 trades
        display_df = results.trades_df.tail(20).copy()
        display_df['entry_time'] = display_df['entry_time'].dt.strftime('%Y-%m-%d %H:%M')
        display_df['exit_time'] = display_df['exit_time'].dt.strftime('%Y-%m-%d %H:%M')
        display_df['result'] = display_df['is_profitable'].apply(lambda x: '✅' if x else '❌')

        st.dataframe(
            display_df[[
                'entry_time', 'exit_time', 'hold_time_hours',
                'pnl_sol', 'pnl_pct', 'result'
            ]].rename(columns={
                'entry_time': 'Entry',
                'exit_time': 'Exit',
                'hold_time_hours': 'Hold (h)',
                'pnl_sol': 'PnL (SOL)',
                'pnl_pct': 'Return %',
                'result': 'Result'
            }),
            use_container_width=True,
            hide_index=True
        )


def display_vs_buy_hold():
    """Compara Follow KOLs vs Buy & Hold"""
    st.markdown("### 🔄 Strategy Comparison")

    # Configuración
    top_n = st.slider("Top N KOLs", 5, 20, 10, key="buy_hold_top_n")

    # Correr backtests
    follow_results = run_follow_kols_backtest(top_n=top_n)
    buy_hold_results = run_buy_hold_backtest(top_n=top_n)

    if not follow_results or not buy_hold_results:
        st.warning("⏳ Not enough data for comparison yet...")
        return

    # Crear comparación
    comparison_data = []

    # Follow KOLs strategy
    comparison_data.append({
        'Strategy': 'Follow KOLs',
        'Total Return (%)': follow_results.total_return,
        'Sharpe Ratio': follow_results.sharpe_ratio,
        'Max Drawdown (%)': abs(follow_results.max_drawdown),
        'Win Rate (%)': follow_results.win_rate * 100,
        'Total Trades': follow_results.total_trades
    })

    # Buy & Hold strategies (mostramos el mejor)
    best_hold_period = max(buy_hold_results.items(), key=lambda x: x[1].sharpe_ratio)
    comparison_data.append({
        'Strategy': f"Buy & Hold ({best_hold_period[0]}",
        'Total Return (%)': best_hold_period[1].total_return,
        'Sharpe Ratio': best_hold_period[1].sharpe_ratio,
        'Max Drawdown (%)': abs(best_hold_period[1].max_drawdown),
        'Win Rate (%)': best_hold_period[1].win_rate * 100,
        'Total Trades': best_hold_period[1].total_trades
    })

    # SOL benchmark (asumimos 0% o se podría calcular)
    comparison_data.append({
        'Strategy': 'SOL Hold (Benchmark)',
        'Total Return (%)': 0,  # TODO: calcular real
        'Sharpe Ratio': 0,
        'Max Drawdown (%)': 0,
        'Win Rate (%)': 0,
        'Total Trades': 1
    })

    comparison_df = pd.DataFrame(comparison_data)

    # Mostrar tabla comparativa
    st.markdown("#### 📊 Strategy Comparison Table")

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )

    # Visualización comparativa
    st.markdown("---")
    st.markdown("#### 📈 Comparison Charts")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            comparison_df,
            x='Strategy',
            y='Total Return (%)',
            title='Total Return Comparison',
            color='Strategy',
            color_discrete_sequence=['#00CC96', '#EF553B', '#636EFA']
        )

        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            comparison_df,
            x='Strategy',
            y='Sharpe Ratio',
            title='Risk-Adjusted Return (Sharpe Ratio)',
            color='Strategy',
            color_discrete_sequence=['#00CC96', '#EF553B', '#636EFA']
        )

        # Add line at 1.0 (good threshold)
        fig.add_hline(y=1.0, line_dash="dash", line_color="green",
                      annotation_text="Good (>1.0)")

        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Conclusión
    st.markdown("---")
    st.markdown("#### 💡 Key Insights")

    if follow_results.total_return > best_hold_period[1].total_return:
        st.success(f"""
        **Follow KOLs strategy outperforms Buy & Hold by {follow_results.total_return - best_hold_period[1].total_return:.1f}%**

        This suggests that following top KOLs provides alpha (excess returns) over simple buy & hold.
        """)
    else:
        st.warning(f"""
        **Buy & Hold outperforms Follow KOLs by {best_hold_period[1].total_return - follow_results.total_return:.1f}%**

        This suggests that timing the market (following KOLs) underperforms simple buy & hold.
        Consider holding positions longer or adjusting KOL selection criteria.
        """)

    if follow_results.sharpe_ratio > 1.0:
        st.success(f"""
        **Excellent Risk-Adjusted Returns**: Sharpe Ratio of {follow_results.sharpe_ratio:.2f} indicates good returns per unit of risk taken.
        """)
    elif follow_results.sharpe_ratio > 0.5:
        st.info(f"""
        **Moderate Risk-Adjusted Returns**: Sharpe Ratio of {follow_results.sharpe_ratio:.2f} shows acceptable returns for the risk level.
        """)


def display_per_kol_analysis():
    """Análisis de performance por KOL individual"""
    st.markdown("### 👤 Per-KOL Performance Analysis")

    try:
        # Leer leaderboard
        with open('data/leaderboard.json', 'r', encoding='utf-8') as f:
            data = json.load(f)

        df = pd.DataFrame(data['leaderboard'])

        # Configuración
        show_n = st.slider("Show Top N KOLs", 5, 50, 20)

        # Filtrar y ordenar
        top_df = df.nlargest(show_n, 'diamond_hand_score')

        # Mostrar tabla
        display_df = top_df[[
            'rank', 'name', 'diamond_hand_score', 'total_trades',
            'total_pnl_sol', 'win_rate', 'avg_hold_time_hours'
        ]].copy()

        display_df['win_rate'] = (display_df['win_rate'] * 100).round(1).astype(str) + '%'
        display_df['avg_hold_time_hours'] = display_df['avg_hold_time_hours'].round(2)

        display_df.columns = [
            'Rank', 'KOL', 'Score', 'Trades',
            'Total PnL (SOL)', 'Win Rate', 'Avg Hold (h)'
        ]

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Visualización
        fig = px.bar(
            top_df,
            x='diamond_hand_score',
            y='name',
            orientation='h',
            title=f'Top {show_n} KOLs by Diamond Hand Score',
            labels={'diamond_hand_score': 'Score', 'name': 'KOL'},
            color='total_pnl_sol',
            color_continuous_scale='Viridis'
        )

        fig.update_yaxes(autorange="reversed")
        fig.update_layout(height=400 + show_n * 15)

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error loading KOL analysis: {e}")


def display_advanced_metrics():
    """Muestra métricas avanzadas de risk management"""
    st.markdown("### 🎯 Advanced Risk Metrics")

    results = run_follow_kols_backtest(top_n=10)

    if not results or results.total_trades == 0:
        st.warning("⏳ Not enough data for advanced metrics...")
        return

    # Risk Metrics Grid
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Return Metrics**")
        st.metric("CAGR", f"{results.cagr:.1f}%", help="Compound Annual Growth Rate")
        st.metric("Avg Trade Return", f"{results.avg_return_per_trade:.2f}%")
        st.metric("Median Return", f"{results.median_return:.2f}%")

    with col2:
        st.markdown("**Risk Metrics**")
        st.metric("Volatility", f"{results.volatility:.1f}%", help="Annualized")
        st.metric("Max Drawdown", f"{results.max_drawdown:.1f}%")
        st.metric("Avg Drawdown", f"{results.avg_drawdown:.1f}%")

    with col3:
        st.markdown("**Risk-Adjusted Returns**")
        st.metric("Sharpe Ratio", f"{results.sharpe_ratio:.2f}")
        st.metric("Sortino Ratio", f"{results.sortino_ratio:.2f}")
        st.metric("Calmar Ratio", f"{results.calmar_ratio:.2f}")

    # Interpretation
    st.markdown("---")
    st.markdown("#### 📖 Metrics Interpretation")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Return Metrics:**")
        st.info("""
        - **CAGR > 50%**: Excellent annual growth
        - **Avg Return > 5%**: Good per-trade average
        - **Median > 0%**: More winners than losers
        """)

    with col2:
        st.markdown("**Risk Metrics:**")
        st.info("""
        - **Volatility < 100%**: Acceptable risk
        - **Max DD < 30%**: Controlled downside
        - **Sharpe > 1.0**: Good risk-adjusted return
        """)

    # Profit Factor & Expectancy
    st.markdown("---")
    st.markdown("#### 💰 Trade Quality Metrics")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Profit Factor",
            f"{results.profit_factor:.2f}",
            help="Gross Profit / Gross Loss (>2 is good)"
        )

        pf_delta = "Excellent" if results.profit_factor >= 2.0 else "Good" if results.profit_factor >= 1.5 else "Poor"
        st.caption(f"Status: {pf_delta}")

    with col2:
        st.metric(
            "Expectancy",
            f"{results.expectancy:.2f}%",
            help="Average profit per trade"
        )

        exp_delta = "Positive" if results.expectancy > 0 else "Negative"
        st.caption(f"Status: {exp_delta}")


def display_performance_dashboard():
    """Dashboard principal de performance analytics"""

    st.markdown("# 📊 Performance Analytics & Backtesting")
    st.markdown("---")

    # Tabs para diferentes secciones
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Model Validation",
        "💰 Follow KOLs",
        "🔄 vs Buy & Hold",
        "👤 Per-KOL Analysis",
        "🎯 Advanced Metrics"
    ])

    with tab1:
        display_model_accuracy()

    with tab2:
        display_follow_kols_backtest()

    with tab3:
        display_vs_buy_hold()

    with tab4:
        display_per_kol_analysis()

    with tab5:
        display_advanced_metrics()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 0.9em;'>
        <p>💡 <strong>Tip</strong>: Revisit this dashboard weekly to track model performance and strategy effectiveness.</p>
        <p>📊 Metrics are updated automatically as new closed positions become available.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    display_performance_dashboard()
