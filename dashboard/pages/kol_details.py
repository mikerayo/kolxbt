"""
KOL Details Page - Advanced Analytics

Shows comprehensive analysis for selected KOLs including:
- Advanced trading performance
- Temporal patterns
- Behavioral analysis
- Current portfolio
- Trade quality metrics
"""

import streamlit as st
from datetime import datetime
import pandas as pd
from dashboard.core.state_manager import get_state
from dashboard.core.kol_analyzer import get_advanced_analyzer
from database import db, KOL


def render_kol_selector():
    """Render KOL selection interface"""
    st.subheader("🔍 Select KOL to Analyze")

    session = db.get_session()

    # Get all KOLs
    kols = session.query(KOL.id, KOL.name).order_by(KOL.name).all()

    if not kols:
        st.warning("No KOLs found in database")
        return None

    # Create dropdown with search
    kol_names = {f"{kol.name} (ID: {kol.id})": kol.id for kol, _ in kols}
    selected = st.selectbox(
        "Choose KOL",
        options=list(kol_names.keys()),
        index=0
    )

    return kol_names[selected]


def render_trading_performance(performance: dict, kol_name: str):
    """Render advanced trading performance metrics"""
    st.subheader("📊 Advanced Trading Performance")

    if not performance:
        st.info("No trading data available")
        return

    # Key metrics in columns
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Biggest Win", f"{performance['biggest_win_sol']:.2f} SOL")

    with c2:
        st.metric("Biggest Loss", f"{performance['biggest_loss_sol']:.2f} SOL")

    with c3:
        st.metric("Sharpe Ratio", f"{performance['sharpe_ratio']:.2f}")

    with c4:
        st.metric("Max Drawdown", f"{performance['max_drawdown_sol']:.2f} SOL")

    st.markdown("---")

    # Detailed metrics
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 📈 Position Metrics")
        st.write(f"**Avg Position Size:** {performance['avg_position_size_sol']:.2f} SOL")
        st.write(f"**Avg Winning Trade:** {performance['avg_winning_trade_sol']:.2f} SOL")
        st.write(f"**Avg Losing Trade:** {performance['avg_losing_trade_sol']:.2f} SOL")
        st.write(f"**Win/Loss Ratio:** {performance['win_loss_ratio']:.2f}")
        st.write(f"**Profit Factor:** {performance['profit_factor']:.2f}")

    with col2:
        st.markdown("##### 🔥 Streak Analysis")
        streak_type = performance['current_streak_type'] or 'none'
        streak_emoji = "🟢" if streak_type == 'win' else "🔴" if streak_type == 'loss' else "⚪"
        st.write(f"**Current Streak:** {streak_emoji} {performance['current_streak']} {streak_type.upper()}s")


def render_temporal_patterns(patterns: dict):
    """Render temporal trading patterns"""
    st.subheader("⏰ Temporal Patterns")

    if not patterns:
        st.info("No temporal data available")
        return

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Best Hour", f"{patterns['best_hour']}:00")

    with c2:
        st.metric("Most Active Hour", f"{patterns['most_active_hour']}:00")

    with c3:
        st.metric("Best Weekday", f"{['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][patterns['best_weekday']]}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 📅 Activity Patterns")
        st.write(f"**Trades per Day:** {patterns['trades_per_day']:.2f}")
        st.write(f"**Avg Time Between Trades:** {patterns['avg_time_between_trades_minutes']:.1f} minutes")
        st.write(f"**Best Hour PnL:** {patterns['best_hour_pnl_sol']:.2f} SOL at {patterns['best_hour_for_pnl']}:00")


def render_behavioral_analysis(behavior: dict):
    """Render behavioral analysis"""
    st.subheader("🎯 Behavioral Analysis")

    if not behavior:
        st.info("No behavioral data available")
        return

    # Metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Buy/Sell Ratio", f"{behavior['buy_sell_ratio']:.2f}")

    with col2:
        st.metric("Avg Flip Ratio", f"{behavior['avg_flip_ratio']:.2f}")

    with col3:
        st.metric("FOMO Score", f"{behavior['fomo_score']:.0f}/100")

    st.markdown("---")

    # Favorite tokens
    st.markdown("##### 🪙 Favorite Tokens")
    if behavior['favorite_tokens']:
        for i, token in enumerate(behavior['favorite_tokens'][:5], 1):
            st.write(f"**{i}.** `{token['address']}` - {token['count']} trades")
    else:
        st.write("No token data available")

    # DEX preference
    st.markdown("##### 🏛️ DEX Preference")
    if behavior['dex_preference']:
        for dex, count in behavior['dex_preference'].items():
            st.write(f"**{dex or 'Unknown'}:** {count} trades")


def render_current_portfolio(portfolio: dict):
    """Render current portfolio status"""
    st.subheader("💼 Current Portfolio")

    if not portfolio or not portfolio.get('has_open_positions'):
        st.info("No open positions")
        return

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Value", f"{portfolio['total_value_sol']:.2f} SOL")

    with col2:
        st.metric("Unrealized PnL", f"{portfolio['total_unrealized_pnl_sol']:+.2f} SOL")

    with col3:
        st.metric("Positions", f"{portfolio['num_positions']}")

    with col4:
        st.metric("Top 3 Concentration", f"{portfolio['top3_concentration_percentage']:.1f}%")

    st.markdown("---")

    # Individual positions
    st.markdown("##### 📋 Open Positions")

    for pos in portfolio['positions']:
        with st.expander(f"{'🟢' if pos['unrealized_pnl_sol'] > 0 else '🔴'} {pos['token_address']} - {pos['unrealized_pnl_sol']:+.2f} SOL ({pos['unrealized_pnl_multiple']:.2f}x)"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.write(f"**Entry:** {pos['entry_time'][:16]}")
                st.write(f"**Hold Time:** {pos['hold_time_hours']:.1f}h")

            with col2:
                st.write(f"**Entry Price:** {pos['entry_price']}")
                st.write(f"**Current Price:** {pos['current_price']}")

            with col3:
                st.write(f"**Invested:** {pos['amount_sol']:.2f} SOL")
                st.write(f"**Current Value:** {pos['current_value_sol']:.2f} SOL")
                st.write(f"**Peak Reached:** {pos['peak_multiple_reached']:.2f}x")


def render_trade_quality(quality: dict):
    """Render trade quality metrics"""
    st.subheader("🎲 Trade Quality")

    if not quality:
        st.info("No quality data available")
        return

    col1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Timing Score", f"{quality['timing_score']:.0f}/100")

    with c2:
        st.metric("Avg Hold Time", f"{quality['avg_hold_time_hours']:.2f}h")

    with c3:
        st.write("")
        st.write(f"**Sold Too Early:** {quality['sold_too_early_rate']:.1f}%")
        st.write(f"**Held Too Long:** {quality['held_too_long_rate']:.1f}%")

    st.markdown("---")

    # Best and worst trades
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 🏆 Best Trade")
        if quality.get('best_trade'):
            bt = quality['best_trade']
            st.write(f"**Token:** `{bt['token']}`")
            st.write(f"**PnL:** {bt['pnl_sol']:.2f} SOL")
            st.write(f"**Multiple:** {bt['multiple']:.2f}x")

    with col2:
        st.markdown("##### 💀 Worst Trade")
        if quality.get('worst_trade'):
            wt = quality['worst_trade']
            st.write(f"**Token:** `{wt['token']}`")
            st.write(f"**PnL:** {wt['pnl_sol']:.2f} SOL")
            st.write(f"**Multiple:** {wt['multiple']:.2f}x")


def render_full_analysis(kol_id: int):
    """Render complete KOL analysis"""
    analyzer = get_advanced_analyzer()
    session = db.get_session()

    # Get analysis
    analysis = analyzer.analyze_kol(session, kol_id)

    if 'error' in analysis:
        st.error(f"Error: {analysis['error']}")
        return

    # Header
    st.title(f"👤 {analysis['name']}")
    st.caption(f"Wallet: `{analysis['wallet_address'][:8]}...{analysis['wallet_address'][-8:]}`")

    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Trades", f"{analysis['total_trades']:,}")
    with col2:
        st.metric("First Trade", f"{analysis['first_trade'][:10]}")
    with col3:
        st.metric("Last Trade", f"{analysis['last_trade'][:10]}")
    with col4:
        delta = (datetime.now() - analysis['last_trade']).total_seconds() / 60
        st.metric("Last Activity", f"{delta:.0f} min ago")

    st.markdown("---")

    # Render all sections
    if analysis.get('trading_performance'):
        render_trading_performance(analysis['trading_performance'], analysis['name'])

    if analysis.get('temporal_patterns'):
        render_temporal_patterns(analysis['temporal_patterns'])

    if analysis.get('behavior'):
        render_behavioral_analysis(analysis['behavior'])

    if analysis.get('portfolio'):
        render_current_portfolio(analysis['portfolio'])

    if analysis.get('trade_quality'):
        render_trade_quality(analysis['trade_quality'])


def main():
    """Main entry point"""
    st.title("👤 KOL Advanced Analytics")

    # KOL selector
    kol_id = render_kol_selector()

    if kol_id:
        render_full_analysis(kol_id)
    else:
        st.warning("Select a KOL to view detailed analysis")


if __name__ == "__main__":
    main()
