import sys
#!/usr/bin/env python3
"""
KOL Tracker Dashboard - COMPLETE UNIFIED INTERFACE
Combina todas las features del sistema en una sola página
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import func
from sqlalchemy.orm import selectinload

from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.database import db, KOL, Trade, ClosedPosition, DiscoveredTrader, TokenInfo
from core.feature_engineering import KOLFeatures, PositionMatcher
from core.ml_models import DiamondHandScorer
from hot_kols_scorer import HotKOLsScorer

# Page config
st.set_page_config(
    page_title="KOL Tracker - Complete Dashboard",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .progress-container {
        background: #f0f0f0;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .progress-bar {
        background: linear-gradient(90deg, #4ECDC4, #44A08D);
        height: 30px;
        border-radius: 15px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
        transition: width 0.3s ease;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #4ECDC4;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)


# ==================== DATA LOADING ====================

@st.cache_data(ttl=60)
def load_tracker_progress():
    """Load tracker progress from JSON file"""
    try:
        with open('data/tracking_progress.json', 'r', encoding='utf-8') as f:
            progress = json.load(f)
        return progress
    except:
        return {"completed": [], "last_update": None}


@st.cache_data(ttl=300)
def load_leaderboard():
    """Load leaderboard from JSON file"""
    try:
        with open('data/leaderboard.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        return pd.DataFrame(data['leaderboard'])
    except:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def get_system_stats():
    """Get complete system statistics"""
    session = db.get_session()

    try:
        # KOLs
        total_kols = session.query(KOL).count()

        # Trades
        total_trades = session.query(Trade).count()

        # Closed positions
        total_positions = session.query(ClosedPosition).count()

        # KOLs with trades
        kols_with_trades = session.query(func.count(func.distinct(Trade.kol_id))).scalar()

        # Discovered traders
        discovered_total = session.query(DiscoveredTrader).count()
        discovered_tracking = session.query(DiscoveredTrader).filter(
            DiscoveredTrader.is_tracking == True
        ).count()
        discovered_promoted = session.query(DiscoveredTrader).filter(
            DiscoveredTrader.promoted_to_kol == True
        ).count()

        # Get recent trades with KOL data loaded
        recent_trades = session.query(Trade).options(
            selectinload(Trade.kol)
        ).order_by(Trade.timestamp.desc()).limit(20).all()

        # Convert to dict
        recent_trades_data = []
        for trade in recent_trades:
            recent_trades_data.append({
                'kol_name': trade.kol.name if trade.kol and trade.kol.name else 'Unknown',
                'kol_wallet': trade.kol.wallet_address if trade.kol else 'Unknown',
                'operation': trade.operation,
                'token_address': trade.token_address,
                'amount_sol': trade.amount_sol,
                'amount_token': trade.amount_token,
                'dex': trade.dex,
                'timestamp': trade.timestamp
            })

        return {
            'total_kols': total_kols,
            'total_trades': total_trades,
            'total_positions': total_positions,
            'kols_with_trades': kols_with_trades,
            'discovered_total': discovered_total,
            'discovered_tracking': discovered_tracking,
            'discovered_promoted': discovered_promoted,
            'recent_trades': recent_trades_data
        }
    finally:
        session.close()


# ==================== HELPER FUNCTIONS ====================

@st.cache_data(ttl=300)
def get_token_info_dict():
    """Get all token info as a dict for easy lookup"""
    session = db.get_session()
    try:
        tokens = session.query(TokenInfo).all()
        return {t.token_address: t for t in tokens}
    finally:
        session.close()


def format_token_display(token_address, token_dict=None):
    """Format token for display with name, symbol, and price"""
    if token_dict is None:
        token_dict = get_token_info_dict()

    token_info = token_dict.get(token_address)

    if token_info:
        # Has metadata
        symbol = token_info.symbol or "?"
        name = token_info.name or "Unknown"
        price = token_info.formatted_price

        if token_info.logo_url:
            return f"![{symbol}]({token_info.logo_url}) **{symbol}** - {name}\n{price}"
        else:
            return f"**{symbol}** - {name}\n{price}"
    else:
        # No metadata, show address
        return f"`{token_address[:8]}...{token_address[-6:]}`"


# ==================== DISPLAY FUNCTIONS ====================

def display_progress_bar(progress_data, stats):
    """Display progress bar of scanned KOLs"""
    st.markdown('<div class="progress-container">', unsafe_allow_html=True)

    completed_count = len(progress_data.get('completed', []))
    total_count = stats['total_kols']

    if total_count > 0:
        progress_pct = (completed_count / total_count) * 100
    else:
        progress_pct = 0

    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown(f"""
        <div class="progress-bar" style="width: {min(progress_pct, 100)}%">
            {completed_count}/{total_count} KOLs Scanned ({progress_pct:.1f}%)
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style='text-align: right; font-size: 0.9rem; color: gray;'>
            {total_count - completed_count} remaining
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


def display_top_metrics(stats):
    """Display key metrics at top"""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="💎 Total KOLs",
            value=f"{stats['total_kols']}",
            delta=None
        )

    with col2:
        st.metric(
            label="📊 Total Trades",
            value=f"{stats['total_trades']}",
            delta=None
        )

    with col3:
        st.metric(
            label="🎯 Closed Positions",
            value=f"{stats['total_positions']}",
            delta=None
        )

    with col4:
        st.metric(
            label="🔥 Active Traders",
            value=f"{stats['kols_with_trades']}",
            delta=None
        )


def display_hot_kols():
    """Muestra Hot KOLs Ranking con KOLs clicables"""
    st.markdown("### 🔥 Hot KOLs (Últimas 24h)")
    st.info("💡 Haz click en el nombre de un KOL para ver sus estadísticas detalladas")

    scorer = HotKOLsScorer()
    hot_kols = scorer.get_hot_kols(top_n=20, hours=24)

    if not hot_kols:
        st.warning("No hay KOLs con actividad reciente")
        return

    # Get selected KOL from session state
    if 'selected_hot_kol' not in st.session_state:
        st.session_state.selected_hot_kol = None

    # Display KOLs con botones clicables
    for i, kol in enumerate(hot_kols):
        cols = st.columns([3, 3, 2, 2, 2, 1])

        with cols[0]:
            # Usar un botón con key único
            if st.button(f"👤 {kol['name']}", key=f"kol_btn_{kol['name']}", use_container_width=True):
                st.session_state.selected_hot_kol = kol['name']
                st.rerun()

        with cols[1]:
            st.write(f"`{kol['short_address']}`")

        with cols[2]:
            st.metric("Score", f"{kol['score']:.0f}", label_visibility="collapsed")

        with cols[3]:
            st.write(f"**{kol['breakdown']['recent_trades']}** trades")

        with cols[4]:
            st.write(f"**{round(kol['breakdown']['recent_volume'], 1)}** SOL")

        with cols[5]:
            estado = '🏆 ELITE' if kol['score'] >= 80 else '🔥 HOT' if kol['score'] >= 70 else '✅ GOOD'
            st.write(estado)

        if i < len(hot_kols) - 1:
            st.markdown("---")

    # Mostrar detalles del KOL seleccionado
    if st.session_state.selected_hot_kol:
        st.markdown("---")
        st.markdown(f"### 📊 Detalles: {st.session_state.selected_hot_kol}")

        # Load leaderboard data
        df_leaderboard = load_leaderboard()

        # Check if KOL exists in leaderboard
        if df_leaderboard.empty:
            st.warning("⏳ No hay datos en el leaderboard aún. El analyzer necesita procesar más datos.")
        elif st.session_state.selected_hot_kol not in df_leaderboard['name'].values:
            st.warning(f"⚠️ El KOL '{st.session_state.selected_hot_kol}' aún no tiene estadísticas en el leaderboard.")
            st.info("💡 Usa la tab de 'KOL Details' para ver todos los KOLs disponibles.")
        else:
            kol_data = df_leaderboard[df_leaderboard['name'] == st.session_state.selected_hot_kol].iloc[0]

            # KOL info
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Diamond Hand Score", f"{kol_data['diamond_hand_score']:.1f}/100")

            with col2:
                st.metric("Total Trades", int(kol_data['total_trades']))

            with col3:
                st.metric("Total PnL", f"{kol_data['total_pnl_sol']:.2f} SOL")

            # Tags
            col1, col2, col3 = st.columns(3)

            with col1:
                if kol_data['is_diamond_hand']:
                    st.success("💎 DIAMOND HAND")
                else:
                    st.info("Not Diamond Hand")

            with col2:
                if kol_data['is_scalper']:
                    st.warning("⚡ SCALPER")
                else:
                    st.info("Not Scalper")

            with col3:
                win_rate = kol_data['win_rate'] * 100
                if win_rate >= 50:
                    st.success(f"Win Rate: {win_rate:.1f}%")
                else:
                    st.warning(f"Win Rate: {win_rate:.1f}%")

            # Detailed stats
            st.markdown("#### 📈 Estadísticas Detalladas")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("3x+ Rate", f"{kol_data['three_x_plus_rate']*100:.1f}%")

            with col2:
                st.metric("Avg Hold Time", f"{kol_data['avg_hold_time_hours']:.2f}h")

            with col3:
                st.metric("Median Hold", f"{kol_data['median_hold_time_hours']:.2f}h")

            with col4:
                st.metric("Consistency", f"{kol_data['consistency_score']:.2f}")

            # Trade history
            st.markdown("#### 📜 Historial de Trades Recientes")

            session = db.get_session()
            try:
                kol_obj = session.query(KOL).filter(KOL.name == st.session_state.selected_hot_kol).first()

                if kol_obj:
                    positions = session.query(ClosedPosition).filter(
                        ClosedPosition.kol_id == kol_obj.id
                    ).order_by(ClosedPosition.exit_time.desc()).limit(10).all()

                    if positions:
                        pos_data = []
                        # Get token info dict
                        token_dict = get_token_info_dict()

                        for pos in positions:
                            token_display = format_token_display(pos.token_address, token_dict)
                            pos_data.append({
                                'Token': token_display,
                                'Entry': pos.entry_time.strftime('%Y-%m-%d %H:%M'),
                                'Exit': pos.exit_time.strftime('%Y-%m-%d %H:%M'),
                                'Hold (h)': f"{pos.hold_time_hours:.2f}",
                                'PnL (SOL)': f"{pos.pnl_sol:.2f}",
                                'Multiple': f"{pos.pnl_multiple:.2f}x",
                                'Result': '✅' if pos.is_profitable else '❌'
                            })

                        pos_df = pd.DataFrame(pos_data)
                        st.dataframe(pos_df, width='stretch', hide_index=True)
                    else:
                        st.info("No closed positions yet.")
            finally:
                session.close()

        # Close button
        if st.button("✖️ Cerrar detalles", key="close_hot_kol_details_v2"):
            st.session_state.selected_hot_kol = None
            st.rerun()


def display_diamond_hands_leaderboard(df):
    """Display Diamond Hands leaderboard"""
    st.markdown("### 💎 Diamond Hands Leaderboard")

    if df.empty:
        st.warning("⏳ No data available yet. Waiting for analyzer to process KOLs...")
        st.info("💡 Tip: The analyzer needs at least 5 closed positions (buy+sell) per KOL to generate rankings.")
        return

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        min_score = st.slider("Min Diamond Hand Score", 0, 100, 0, key='dh_min_score')

    with col2:
        min_trades = st.slider("Min Trades", 1, 100, 5, key='dh_min_trades')

    with col3:
        show_only = st.selectbox(
            "Show",
            ["All", "Diamond Hands Only", "Scalpers Only"],
            key='dh_show_only'
        )

    # Apply filters
    filtered = df[df['diamond_hand_score'] >= min_score]
    filtered = filtered[filtered['total_trades'] >= min_trades]

    if show_only == "Diamond Hands Only":
        filtered = filtered[filtered['is_diamond_hand'] == True]
    elif show_only == "Scalpers Only":
        filtered = filtered[filtered['is_scalper'] == True]

    # Display table
    display_df = filtered[[
        'rank', 'name', 'diamond_hand_score', 'total_trades',
        'three_x_plus_rate', 'win_rate', 'avg_hold_time_hours', 'total_pnl_sol'
    ]].copy()

    display_df['three_x_plus_rate'] = (display_df['three_x_plus_rate'] * 100).round(1).astype(str) + '%'
    display_df['win_rate'] = (display_df['win_rate'] * 100).round(1).astype(str) + '%'
    display_df['avg_hold_time_hours'] = display_df['avg_hold_time_hours'].round(2)
    display_df['total_pnl_sol'] = display_df['total_pnl_sol'].round(2)

    display_df.columns = [
        'Rank', 'KOL', 'Score', 'Trades',
        '3x+ Rate', 'Win Rate', 'Avg Hold (h)', 'Total PnL (SOL)'
    ]

    st.dataframe(display_df, width='stretch', hide_index=True)


def display_discovered_traders():
    """Muestra Discovered Traders"""
    st.markdown("### 🕵️ Discovered Traders")

    session = db.get_session()

    try:
        # Get discovered traders
        traders = session.query(DiscoveredTrader).order_by(
            DiscoveredTrader.discovery_score.desc()
        ).limit(20).all()

        if not traders:
            st.info("No traders descubiertos aún. Ejecuta el discovery para encontrar nuevos traders.")
            return

        # Convert to DataFrame
        df_data = []
        for trader in traders:
            status = []
            if trader.promoted_to_kol:
                status.append("PROMOTED")
            elif trader.is_tracking:
                status.append("TRACKING")
            else:
                status.append("DISCOVERED")

            df_data.append({
                'Wallet': trader.short_address,
                'Score': trader.discovery_score,
                'Calidad': trader.quality_label,
                'Trades': trader.total_trades,
                'Win Rate': f"{trader.win_rate:.1%}",
                '3x+ Rate': f"{trader.three_x_rate:.1%}",
                'Estado': ' '.join(status)
            })

        df = pd.DataFrame(df_data)
        st.dataframe(df, width='stretch', hide_index=True)

        st.info("💡 Los traders descubiertos se promocionan automáticamente a KOL si score >= 80")

    finally:
        session.close()


def display_charts(df):
    """Display visualization charts"""
    if df.empty:
        return

    st.markdown("### 📈 Visualizations")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Diamond Hand Score Distribution")
        fig = px.histogram(
            df,
            x='diamond_hand_score',
            nbins=20,
            title="Distribution of Diamond Hand Scores",
            labels={'diamond_hand_score': 'Score'},
            color_discrete_sequence=['#4ECDC4']
        )
        fig.update_layout(bargap=0.1)
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.markdown("#### Win Rate vs 3x+ Rate")
        fig = px.scatter(
            df,
            x='win_rate',
            y='three_x_plus_rate',
            size='total_trades',
            color='diamond_hand_score',
            hover_name='name',
            title="Win Rate vs 3x+ Rate",
            labels={
                'win_rate': 'Win Rate',
                'three_x_plus_rate': '3x+ Rate',
                'total_trades': 'Total Trades'
            },
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, width='stretch')

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Hold Time Distribution")
        fig = px.box(
            df,
            y='avg_hold_time_hours',
            title="Average Hold Time Distribution",
            labels={'avg_hold_time_hours': 'Hours'}
        )
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.markdown("#### Top 10 KOLs by Score")
        top10 = df.nlargest(10, 'diamond_hand_score')
        fig = px.bar(
            top10,
            x='diamond_hand_score',
            y='name',
            orientation='h',
            title="Top 10 Diamond Hands",
            labels={'diamond_hand_score': 'Score', 'name': 'KOL'},
            color='diamond_hand_score',
            color_continuous_scale='Viridis'
        )
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, width='stretch')


def display_recent_trades(stats):
    """Display recent trades"""
    st.markdown("### 🔄 Recent Trades")

    trades = stats['recent_trades']

    if not trades or len(trades) == 0:
        st.warning("⏳ No recent trades available")
        return

    # Get token info dict
    token_dict = get_token_info_dict()

    trade_data = []
    for trade in trades:
        token_display = format_token_display(trade['token_address'], token_dict)
        trade_data.append({
            'KOL': trade['kol_name'],
            'Operation': trade['operation'].upper(),
            'Token': token_display,
            'Amount SOL': f"{trade['amount_sol']:.4f}",
            'Amount Token': f"{trade['amount_token']:.2f}",
            'DEX': trade['dex'] or 'Unknown',
            'Time': trade['timestamp'].strftime('%Y-%m-%d %H:%M')
        })

    df = pd.DataFrame(trade_data)
    st.dataframe(df, width='stretch', hide_index=True)


def display_kol_details(df):
    """Display individual KOL details"""
    st.markdown("### 🔍 KOL Details")

    if df.empty:
        st.warning("⏳ No KOL data available yet")
        return

    # KOL selector
    kol_names = [''] + list(df['name'].unique())
    selected_kol = st.selectbox("Select KOL", kol_names, key='kol_selector')

    if not selected_kol:
        st.info("👆 Select a KOL above to see detailed statistics")
        return

    # Get KOL data
    kol_data = df[df['name'] == selected_kol].iloc[0]

    # Display KOL info
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Diamond Hand Score", f"{kol_data['diamond_hand_score']:.1f}/100")

    with col2:
        st.metric("Total Trades", int(kol_data['total_trades']))

    with col3:
        st.metric("Total PnL", f"{kol_data['total_pnl_sol']:.2f} SOL")

    # Tags
    col1, col2, col3 = st.columns(3)

    with col1:
        if kol_data['is_diamond_hand']:
            st.success("💎 DIAMOND HAND")
        else:
            st.info("Not Diamond Hand")

    with col2:
        if kol_data['is_scalper']:
            st.warning("⚡ SCALPER")
        else:
            st.info("Not Scalper")

    with col3:
        win_rate = kol_data['win_rate'] * 100
        if win_rate >= 50:
            st.success(f"Win Rate: {win_rate:.1f}%")
        else:
            st.warning(f"Win Rate: {win_rate:.1f}%")

    # Detailed stats
    st.markdown("#### 📊 Detailed Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("3x+ Rate", f"{kol_data['three_x_plus_rate']*100:.1f}%")

    with col2:
        st.metric("Avg Hold Time", f"{kol_data['avg_hold_time_hours']:.2f}h")

    with col3:
        st.metric("Median Hold", f"{kol_data['median_hold_time_hours']:.2f}h")

    with col4:
        st.metric("Consistency", f"{kol_data['consistency_score']:.2f}")

    # Trade history
    st.markdown("#### 📜 Trade History")

    session = db.get_session()
    try:
        # Get KOL
        kol = session.query(KOL).filter(KOL.name == selected_kol).first()

        if kol:
            # Get closed positions
            positions = session.query(ClosedPosition).filter(
                ClosedPosition.kol_id == kol.id
            ).order_by(ClosedPosition.exit_time.desc()).limit(20).all()

            if positions:
                pos_data = []
                # Get token info dict
                token_dict = get_token_info_dict()

                for pos in positions:
                    token_display = format_token_display(pos.token_address, token_dict)
                    pos_data.append({
                        'Token': token_display,
                        'Entry': pos.entry_time.strftime('%Y-%m-%d %H:%M'),
                        'Exit': pos.exit_time.strftime('%Y-%m-%d %H:%M'),
                        'Hold (h)': f"{pos.hold_time_hours:.2f}",
                        'PnL (SOL)': f"{pos.pnl_sol:.2f}",
                        'Multiple': f"{pos.pnl_multiple:.2f}x",
                        'Result': '✅ Profit' if pos.is_profitable else '❌ Loss'
                    })

                pos_df = pd.DataFrame(pos_data)
                st.dataframe(pos_df, width='stretch', hide_index=True)
            else:
                st.info("No closed positions yet. Positions need both buy and sell trades.")
        else:
            st.warning("KOL not found in database")
    finally:
        session.close()


def display_system_overview(stats):
    """Muestra overview completo del sistema"""
    st.markdown("### 📊 System Overview")

    # KOLs Metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">KOLs Trackeados</div>
        </div>
        """.format(stats['total_kols']), unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{:,}</div>
            <div class="metric-label">Total Trades</div>
        </div>
        """.format(stats['total_trades']), unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{:,}</div>
            <div class="metric-label">Posiciones Cerradas</div>
        </div>
        """.format(stats['total_positions']), unsafe_allow_html=True)

    # Discovery stats
    st.markdown("---")
    st.markdown("#### 🆕 Discovery Stats")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Descubiertos", stats['discovered_total'])

    with col2:
        st.metric("En Tracking", stats['discovered_tracking'])

    with col3:
        st.metric("Promovidos a KOL", stats['discovered_promoted'])

    # Background processes status
    st.markdown("---")
    st.markdown("#### ⚙️ Background Processes")

    st.info("""
    **Procesos Activos:**
    - 🔍 **Tracker**: Escaneando trades cada 5 minutos
    - 🧠 **ML Trainer**: Reentrenando cada 1 hora
    - 🕵️ **Discovery**: Buscando nuevas wallets cada 6 horas
    - 🪙 **Token Updater**: Actualizando metadata cada 35 minutos
    """)


def display_tokens():
    """Muestra los tokens trackeados con su metadata de DexScreener"""
    st.markdown("### 🪙 Tracked Tokens")

    session = db.get_session()

    try:
        # Get all tokens with info
        tokens = session.query(TokenInfo).order_by(
            TokenInfo.last_updated.desc()
        ).all()

        if not tokens:
            st.warning("⏳ No hay tokens con metadata aún. El actualizador está corriendo en background...")
            st.info("💡 Los tokens se actualizan automáticamente cada 35 minutos desde DexScreener API")
            return

        # Stats
        st.markdown("#### 📊 Token Stats")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Tokens", len(tokens))

        with col2:
            with_price = len([t for t in tokens if t.price_usd])
            st.metric("Con Precio", with_price)

        with col3:
            with_logo = len([t for t in tokens if t.logo_url])
            st.metric("Con Logo", with_logo)

        with col4:
            # Total liquidity
            total_liq = sum([t.liquidity_usd or 0 for t in tokens])
            st.metric("Liquidez Total", f"${total_liq:,.0f}")

        st.markdown("---")

        # ==================== DASHBOARD INTERACTIVO ====================
        st.markdown("#### 📈 Token Analytics Dashboard")

        # Create tabs for different visualizations
        analytics_tab1, analytics_tab2, analytics_tab3 = st.tabs([
            "🏆 Top Tokens",
            "📊 Distribution",
            "🔍 Correlations"
        ])

        # Prepare data for visualizations
        tokens_with_liq = [t for t in tokens if t.liquidity_usd and t.liquidity_usd > 0]
        tokens_with_volume = [t for t in tokens if t.volume_24h_usd and t.volume_24h_usd > 0]
        tokens_with_change = [t for t in tokens if t.change_24h_percent is not None]
        tokens_with_price = [t for t in tokens if t.price_usd and t.price_usd > 0]

        with analytics_tab1:
            col1, col2 = st.columns(2)

            # Top 10 by Liquidity
            with col1:
                st.markdown("##### 💰 Top 10 Tokens by Liquidity")
                if tokens_with_liq:
                    top_liq = sorted(tokens_with_liq, key=lambda t: t.liquidity_usd, reverse=True)[:10]
                    liq_data = []
                    for t in top_liq:
                        liq_data.append({
                            'Token': t.symbol or t.token_address[:8],
                            'Liquidity (USD)': t.liquidity_usd,
                            'Name': t.name or 'Unknown'
                        })

                    liq_df = pd.DataFrame(liq_data)
                    fig = px.bar(
                        liq_df,
                        x='Liquidity (USD)',
                        y='Token',
                        orientation='h',
                        color='Liquidity (USD)',
                        color_continuous_scale='Viridis',
                        title='Top 10 by Liquidity',
                        hover_data=['Name']
                    )
                    fig.update_yaxes(autorange="reversed")
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No liquidity data available")

            # Top 10 by Volume 24h
            with col2:
                st.markdown("##### 📊 Top 10 Tokens by Volume 24h")
                if tokens_with_volume:
                    top_vol = sorted(tokens_with_volume, key=lambda t: t.volume_24h_usd, reverse=True)[:10]
                    vol_data = []
                    for t in top_vol:
                        vol_data.append({
                            'Token': t.symbol or t.token_address[:8],
                            'Volume 24h (USD)': t.volume_24h_usd,
                            'Name': t.name or 'Unknown'
                        })

                    vol_df = pd.DataFrame(vol_data)
                    fig = px.bar(
                        vol_df,
                        x='Volume 24h (USD)',
                        y='Token',
                        orientation='h',
                        color='Volume 24h (USD)',
                        color_continuous_scale='Plasma',
                        title='Top 10 by Volume 24h',
                        hover_data=['Name']
                    )
                    fig.update_yaxes(autorange="reversed")
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No volume data available")

            # Winners and Losers 24h
            st.markdown("##### 🚀 Biggest Movers (24h)")

            if tokens_with_change:
                col_a, col_b = st.columns(2)

                # Top gainers
                with col_a:
                    st.markdown("**🟢 Top 5 Gainers**")
                    gainers = sorted(tokens_with_change, key=lambda t: t.change_24h_percent, reverse=True)[:5]
                    for i, t in enumerate(gainers, 1):
                        st.metric(
                            f"{i}. {t.symbol or t.token_address[:8]}",
                            f"+{t.change_24h_percent:.2f}%",
                            f"${t.price_usd:.8f}" if t.price_usd else "N/A"
                        )

                # Top losers
                with col_b:
                    st.markdown("**🔴 Top 5 Losers**")
                    losers = sorted(tokens_with_change, key=lambda t: t.change_24h_percent)[:5]
                    for i, t in enumerate(losers, 1):
                        st.metric(
                            f"{i}. {t.symbol or t.token_address[:8]}",
                            f"{t.change_24h_percent:.2f}%",
                            f"${t.price_usd:.8f}" if t.price_usd else "N/A"
                        )

                # Bar chart of all tokens with 24h change
                st.markdown("**Price Change 24h Distribution**")
                change_data = []
                for t in tokens_with_change[:20]:  # Top 20 for visualization
                    change_data.append({
                        'Token': t.symbol or t.token_address[:8],
                        'Change %': t.change_24h_percent,
                        'Type': 'Positive' if t.change_24h_percent >= 0 else 'Negative'
                    })

                change_df = pd.DataFrame(change_data)
                fig = px.bar(
                    change_df,
                    x='Token',
                    y='Change %',
                    color='Type',
                    color_discrete_map={'Positive': '#00CC96', 'Negative': '#EF553B'},
                    title='Price Change 24h (%)'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No 24h change data available")

        with analytics_tab2:
            col1, col2 = st.columns(2)

            # DEX Distribution
            with col1:
                st.markdown("##### 🏦 DEX Distribution")
                dex_counts = {}
                for t in tokens:
                    dex = t.dex_id or 'Unknown'
                    dex_counts[dex] = dex_counts.get(dex, 0) + 1

                if dex_counts:
                    dex_df = pd.DataFrame([
                        {'DEX': dex, 'Count': count}
                        for dex, count in sorted(dex_counts.items(), key=lambda x: x[1], reverse=True)
                    ])

                    fig = px.pie(
                        dex_df,
                        values='Count',
                        names='DEX',
                        title='Tokens by DEX',
                        hole=0.3
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No DEX data available")

            # Price Distribution
            with col2:
                st.markdown("##### 💵 Price Distribution")
                if tokens_with_price:
                    price_data = [t.price_usd for t in tokens_with_price]

                    fig = px.histogram(
                        x=price_data,
                        nbins=30,
                        title='Price Distribution (USD)',
                        labels={'x': 'Price (USD)', 'y': 'Count'}
                    )
                    fig.update_layout(height=400)
                    fig.update_xaxes(type="log")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No price data available")

            # Liquidity Distribution
            st.markdown("##### 💧 Liquidity Distribution")
            if tokens_with_liq:
                liq_values = [t.liquidity_usd for t in tokens_with_liq]

                fig = px.box(
                    x=liq_values,
                    title='Liquidity Distribution (USD)',
                    labels={'x': 'Liquidity (USD)'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

                # Statistics
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    st.metric("Min", f"${min(liq_values):,.0f}")
                with col_b:
                    st.metric("Max", f"${max(liq_values):,.0f}")
                with col_c:
                    st.metric("Mean", f"${sum(liq_values)/len(liq_values):,.0f}")
                with col_d:
                    st.metric("Median", f"${sorted(liq_values)[len(liq_values)//2]:,.0f}")
            else:
                st.info("No liquidity data available")

        with analytics_tab3:
            # Price vs Volume Scatter
            st.markdown("##### 📊 Price vs Volume Correlation")

            tokens_price_vol = [t for t in tokens if t.price_usd and t.volume_24h_usd]
            if tokens_price_vol:
                pv_data = []
                for t in tokens_price_vol:
                    pv_data.append({
                        'Token': t.symbol or t.token_address[:8],
                        'Price (USD)': t.price_usd,
                        'Volume 24h (USD)': t.volume_24h_usd,
                        'Liquidity (USD)': t.liquidity_usd or 0,
                        'Change 24h (%)': t.change_24h_percent or 0
                    })

                pv_df = pd.DataFrame(pv_data)

                fig = px.scatter(
                    pv_df,
                    x='Price (USD)',
                    y='Volume 24h (USD)',
                    color='Liquidity (USD)',
                    size='Liquidity (USD)',
                    hover_data=['Token'],
                    title='Price vs Volume (colored by Liquidity)',
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(height=500)
                fig.update_xaxes(type="log")
                fig.update_yaxes(type="log")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough data for price vs volume analysis")

            # Liquidity vs Volume Scatter
            st.markdown("##### 💧📊 Liquidity vs Volume Correlation")

            tokens_liq_vol = [t for t in tokens if t.liquidity_usd and t.volume_24h_usd]
            if tokens_liq_vol:
                lv_data = []
                for t in tokens_liq_vol:
                    lv_data.append({
                        'Token': t.symbol or t.token_address[:8],
                        'Liquidity (USD)': t.liquidity_usd,
                        'Volume 24h (USD)': t.volume_24h_usd,
                        'Change 24h (%)': t.change_24h_percent or 0
                    })

                lv_df = pd.DataFrame(lv_data)

                fig = px.scatter(
                    lv_df,
                    x='Liquidity (USD)',
                    y='Volume 24h (USD)',
                    color='Change 24h (%)',
                    size='Volume 24h (USD)',
                    hover_data=['Token'],
                    title='Liquidity vs Volume (colored by 24h Change)',
                    color_continuous_scale='RdYlGn',
                    range_color=[-20, 20]
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough data for liquidity vs volume analysis")

        st.markdown("---")

        # ==================== FILTROS ====================
        st.markdown("#### 🔍 Token Explorer")

        # Filters
        col1, col2, col3 = st.columns(3)

        with col1:
            search = st.text_input("🔍 Search", placeholder="Symbol or name...", key="token_search")

        with col2:
            min_liquidity = st.number_input("💰 Min Liquidity (USD)", min_value=0, value=0, step=10000, key="token_min_liq")

        with col3:
            sort_by = st.selectbox("📶 Sort By", [
                "Last Updated",
                "Liquidity (High to Low)",
                "Volume 24h (High to Low)",
                "Price Change 24h (High to Low)",
                "Price (High to Low)",
                "Price (Low to High)"
            ], key="token_sort")

        # Apply filters
        filtered_tokens = []

        for token in tokens:
            # Search filter
            if search:
                search_lower = search.lower()
                symbol = (token.symbol or "").lower()
                name = (token.name or "").lower()
                if search_lower not in symbol and search_lower not in name:
                    continue

            # Liquidity filter
            if min_liquidity > 0:
                liq = token.liquidity_usd or 0
                if liq < min_liquidity:
                    continue

            filtered_tokens.append(token)

        # Sort
        if sort_by == "Liquidity (High to Low)":
            filtered_tokens.sort(key=lambda t: t.liquidity_usd or 0, reverse=True)
        elif sort_by == "Volume 24h (High to Low)":
            filtered_tokens.sort(key=lambda t: t.volume_24h_usd or 0, reverse=True)
        elif sort_by == "Price Change 24h (High to Low)":
            filtered_tokens.sort(key=lambda t: t.change_24h_percent or 0, reverse=True)
        elif sort_by == "Price (High to Low)":
            filtered_tokens.sort(key=lambda t: t.price_usd or 0, reverse=True)
        elif sort_by == "Price (Low to High)":
            filtered_tokens.sort(key=lambda t: t.price_usd or float('inf'))
        else:  # Last Updated
            pass  # Already sorted by last_updated

        # Display tokens
        st.markdown(f"#### 📋 Showing {len(filtered_tokens)} tokens")

        # Create table data
        token_data = []
        for token in filtered_tokens[:100]:  # Limit to 100 for display
            # Logo + Symbol + Name
            if token.logo_url:
                token_display = f"![{token.symbol or '?'}]({token.logo_url}) **{token.symbol or '?'}**\n{token.name or 'Unknown'}"
            else:
                token_display = f"**{token.symbol or '?'}**\n{token.name or 'Unknown'}"

            # Price with formatting
            price_col = token.formatted_price

            # Change 24h with color
            change = token.change_24h_percent
            if change is not None:
                if change >= 0:
                    change_col = f"🟢 +{change:.2f}%"
                else:
                    change_col = f"🔴 {change:.2f}%"
            else:
                change_col = "N/A"

            token_data.append({
                'Token': token_display,
                'Price': price_col,
                'Change 24h': change_col,
                'Liquidity': token.formatted_liquidity,
                'Volume 24h': token.formatted_volume,
                'DEX': f"{token.dex_id or '?'} ({token.chain_id or 'solana'})",
                'Last Updated': token.last_updated.strftime('%Y-%m-%d %H:%M') if token.last_updated else 'N/A'
            })

        if token_data:
            df = pd.DataFrame(token_data)
            st.dataframe(df, width='stretch', hide_index=True, use_container_width=True)
        else:
            st.info("No tokens match the filters")

        # Detailed view
        if filtered_tokens:
            st.markdown("---")
            st.markdown("#### 🔍 Token Details")

            selected_token = st.selectbox(
                "Select token for details",
                options=filtered_tokens,
                format_func=lambda t: f"{t.symbol or t.token_address[:8]} - {t.name or 'Unknown'}",
                key="token_detail_selector"
            )

            if selected_token:
                col1, col2 = st.columns([1, 2])

                with col1:
                    st.markdown("##### Token Info")
                    if selected_token.logo_url:
                        st.image(selected_token.logo_url, width=100)

                    st.markdown(f"""
                    **Symbol:** {selected_token.symbol or 'N/A'}

                    **Name:** {selected_token.name or 'N/A'}

                    **Address:** `{selected_token.token_address}`

                    **Chain:** {selected_token.chain_id or 'solana'}

                    **DEX:** {selected_token.dex_id or 'N/A'}

                    **Pair Address:** `{selected_token.pair_address or 'N/A'}`
                    """)

                with col2:
                    st.markdown("##### Market Data")

                    # Price info
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Price", selected_token.formatted_price)
                    with col_b:
                        if selected_token.change_24h_percent is not None:
                            st.metric("Change 24h", f"{selected_token.change_24h_percent:.2f}%")

                    # Market cap and liquidity
                    col_c, col_d = st.columns(2)
                    with col_c:
                        if selected_token.fdv_usd:
                            st.metric("FDV", f"${selected_token.fdv_usd:,.0f}")
                    with col_d:
                        if selected_token.liquidity_usd:
                            st.metric("Liquidity", selected_token.formatted_liquidity)

                    # Volume
                    if selected_token.volume_24h_usd:
                        st.metric("Volume 24h", selected_token.formatted_volume)

                    st.markdown(f"**Last Updated:** {selected_token.last_updated.strftime('%Y-%m-%d %H:%M:%S') if selected_token.last_updated else 'N/A'}")

    finally:
        session.close()


# ==================== MAIN ====================

def main():
    """Main unified dashboard"""

    # Header
    st.markdown('<h1 class="main-header">💎 KOL Tracker - Unified Dashboard</h1>', unsafe_allow_html=True)

    # Load data
    progress_data = load_tracker_progress()
    df_leaderboard = load_leaderboard()
    stats = get_system_stats()

    # Display progress bar
    display_progress_bar(progress_data, stats)

    # Sidebar
    st.sidebar.markdown("### ⚙️ Settings")
    auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (30s)", value=False, key="auto_refresh_main")
    st.sidebar.caption(f"Last update: {datetime.now().strftime('%H:%M:%S')}")

    # Top metrics
    display_top_metrics(stats)

    st.markdown("---")

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "🔥 Hot KOLs",
        "💎 Diamond Hands",
        "🕵️ Discovered",
        "📈 Gráficos",
        "🔄 Recent Trades",
        "🔍 KOL Details",
        "🪙 Tokens",
        "📊 System Overview"
    ])

    with tab1:
        display_hot_kols()

    with tab2:
        display_diamond_hands_leaderboard(df_leaderboard)

    with tab3:
        display_discovered_traders()

    with tab4:
        display_charts(df_leaderboard)

    with tab5:
        display_recent_trades(stats)

    with tab6:
        display_kol_details(df_leaderboard)

    with tab7:
        display_tokens()

    with tab8:
        display_system_overview(stats)

    # Footer
    st.markdown("---")
    st.markdown(
        f"""
        <div style='text-align: center; color: gray;'>
            <p>KOL Tracker ML System - Tracking {stats['total_kols']} KOLs</p>
            <p>Data updated every 5 minutes • ML retraining every 1 hour • Discovery every 6 hours</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Auto-refresh
    if auto_refresh:
        import time
        time.sleep(30)
        st.rerun()


if __name__ == "__main__":
    main()
