"""
KOL Tracker ML - Dashboard v2.1

Enhanced modular dashboard with:
- Advanced analytics
- ML predictions
- Fuzzy search
- Animated UI/UX
- Tooltips & Help system
- Responsive design

Usage:
    streamlit run dashboard/app.py

    or with custom options:
    streamlit run dashboard/app.py --server.headless true --server.port 8502
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dashboard.core.state_manager import get_state
from dashboard.core.data_manager import get_data_manager
from dashboard.components.search import render_search_bar, fuzzy_search_component
from dashboard.styles.theme import render_theme, apply_theme_from_state, create_metric_card
from dashboard.components.tooltips import render_tooltip


# Page config
st.set_page_config(
    page_title="KOL Tracker ML v2.1",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Don't apply theme at module level - will be applied in main() after state init


def render_header():
    """Render dashboard header with enhanced styling"""
    st.markdown('<div class="gradient-text fade-in">💎 KOL Tracker ML v2.1</div>', unsafe_allow_html=True)

    # Add info expander with glossary
    with st.expander("💡 Quick Guide & Glossary", expanded=False):
        st.markdown("""
        ### Welcome to KOL Tracker ML!

        **This dashboard helps you:**
        - 🎯 Track Key Opinion Leaders (KOLs) trading on Solana
        - 📊 Analyze their performance with advanced metrics
        - 🤖 Use ML predictions to identify opportunities
        - 🪙 Research tokens and trading patterns

        **Quick Start:**
        1. Browse the **Leaderboard** to find top-performing KOLs
        2. Use **Analytics** to understand trends and patterns
        3. Check **ML Predictions** for probability-based insights
        4. **Compare** KOLs to find the best performers

        **Key Metrics Explained:**
        - **Diamond Hand Score**: Overall quality (0-100, higher is better)
        - **3x+ Rate**: % of positions that reached 3x or more
        - **Win Rate**: % of profitable trades
        - **Avg Hold Time**: How long they typically hold positions

        🔍 For detailed explanations, check the tooltips (ℹ️) throughout the dashboard!
        """)

    st.markdown("---")


def render_metrics():
    """Render key metrics with tooltips and enhanced styling"""
    data_manager = get_data_manager()

    # Get stats
    stats = data_manager.get_database_stats()

    st.markdown('<h2 style="margin-bottom: 1.5rem;">📊 Dashboard Overview</h2>', unsafe_allow_html=True)

    # Create enhanced metric cards with tooltips
    col1, col2, col3, col4 = st.columns(4)

    # Metric card 1
    with col1:
        st.markdown("""
        <div class="metric-card fade-in">
            <div style="display: flex; align-items: center; margin-bottom: 0.75rem;">
                <div style="font-size: 2rem; margin-right: 0.75rem;">💎</div>
                <div style="font-size: 0.75rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;">
                    Total KOLs
                </div>
            </div>
            <div class="metric-value" style="color: var(--accent-primary);">{:,}</div>
        </div>
        """.format(stats['total_kols']), unsafe_allow_html=True)
        with st.expander("ℹ️"):
            st.markdown("""
            **Total number of tracked KOLs**

            Key Opinion Leaders whose wallets we monitor for trading activity.
            """)

    # Metric card 2
    with col2:
        st.markdown("""
        <div class="metric-card fade-in" style="animation-delay: 0.1s;">
            <div style="display: flex; align-items: center; margin-bottom: 0.75rem;">
                <div style="font-size: 2rem; margin-right: 0.75rem;">📊</div>
                <div style="font-size: 0.75rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;">
                    Total Trades
                </div>
            </div>
            <div class="metric-value" style="color: var(--accent-secondary);">{:,}</div>
        </div>
        """.format(stats['total_trades']), unsafe_allow_html=True)
        with st.expander("ℹ️"):
            st.markdown("""
            **Total number of trades tracked**

            All buy/sell operations across all DEXs.
            """)

    # Metric card 3
    with col3:
        st.markdown("""
        <div class="metric-card fade-in" style="animation-delay: 0.2s;">
            <div style="display: flex; align-items: center; margin-bottom: 0.75rem;">
                <div style="font-size: 2rem; margin-right: 0.75rem;">🎯</div>
                <div style="font-size: 0.75rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;">
                    Closed Positions
                </div>
            </div>
            <div class="metric-value" style="color: var(--accent-success);">{:,}</div>
        </div>
        """.format(stats['total_positions']), unsafe_allow_html=True)
        with st.expander("ℹ️"):
            st.markdown("""
            **Completed trading positions**

            Positions with both buy AND sell detected.
            Required for performance analysis.
            """)

    # Metric card 4
    with col4:
        st.markdown("""
        <div class="metric-card fade-in" style="animation-delay: 0.3s;">
            <div style="display: flex; align-items: center; margin-bottom: 0.75rem;">
                <div style="font-size: 2rem; margin-right: 0.75rem;">🔥</div>
                <div style="font-size: 0.75rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;">
                    Active Traders
                </div>
            </div>
            <div class="metric-value" style="color: var(--accent-warning);">{:,}</div>
        </div>
        """.format(stats['kols_with_trades']), unsafe_allow_html=True)
        with st.expander("ℹ️"):
            st.markdown("""
            **KOLs with trading activity**

            KOLs that have executed at least one trade.
            """)

    st.markdown('<hr style="margin: 2rem 0; border: none; height: 1px; background: linear-gradient(90deg, transparent, var(--border-color), transparent);">', unsafe_allow_html=True)


def render_sidebar():
    """Render sidebar with navigation, filters, and settings"""
    state = get_state()

    with st.sidebar:
        st.markdown("### ⚙️ Settings & Preferences")

        # Theme selector with tooltip
        theme = st.selectbox(
            "🎨 Theme",
            ['dark', 'light', 'cyberpunk'],
            index=['dark', 'light', 'cyberpunk'].index(
                state.get_preference('theme', 'dark')
            ),
            key='theme_selector',
            help="Choose your preferred color scheme"
        )
        state.set_preference('theme', theme)

        st.markdown("---")

        # Auto-refresh with explanation
        st.markdown("#### 🔄 Auto-refresh")
        auto_refresh = st.checkbox(
            "Enable auto-refresh",
            value=state.get_preference('auto_refresh', False),
            key='auto_refresh',
            help="Automatically reload data at regular intervals"
        )
        state.set_preference('auto_refresh', auto_refresh)

        if auto_refresh:
            interval = st.slider(
                "Refresh interval (seconds)",
                10, 300,
                state.get_preference('refresh_interval', 30),
                key='refresh_interval',
                help="How often to refresh data (lower = more frequent updates)"
            )
            state.set_preference('refresh_interval', interval)

            st.info(f"🔄 Dashboard will refresh every {interval} seconds")

        st.markdown("---")

        # Data freshness indicator
        st.markdown("#### 📊 Data Status")
        data_age = state.get_data_age_seconds()
        if data_age is not None:
            if data_age < 60:
                freshness = "🟢 Fresh (< 1 min)"
                color = "success"
            elif data_age < 300:
                freshness = "🟡 5 min ago"
                color = "warning"
            else:
                freshness = "🔴 Stale (> 5 min)"
                color = "error"

            st.markdown(f"**Freshness:** {freshness}")
            st.caption(f"Last update: {int(data_age)}s ago")

            if color == "warning":
                st.warning("💡 Data is getting stale. Consider refreshing.")
            elif color == "error":
                st.error("⚠️ Data is stale! Refresh recommended.")
        else:
            st.caption("ℹ️ Data freshness not available")

        st.markdown("---")

        # Tracker status
        st.markdown("#### 🔍 Tracker Status")
        st.markdown("""
        <div class='info-box'>
            <strong>Status:</strong> 🟢 Running<br/>
            <strong>Mode:</strong> Continuous<br/>
            <strong>Scan:</strong> Every 5 min
        </div>
        """, unsafe_allow_html=True)

        st.caption("ℹ️ Tracker is continuously scanning for new activity")

        st.markdown("---")

        # System info
        st.markdown("#### ℹ️ System Info")
        st.markdown("""
        **Version:** v2.1 (Enhanced)

        **Features:**
        - ✅ Fuzzy Search
        - ✅ ML Predictions
        - ✅ Advanced Analytics
        - ✅ Tooltips & Help
        - ✅ Responsive Design
        - ✅ Animations
        """)

        # Feedback link
        st.markdown("---")
        st.markdown(
            "<div style='text-align: center; font-size: 0.875rem;'>"
            "💡 Need help? Check the Quick Guide above"
            "</div>",
            unsafe_allow_html=True
        )


def render_leaderboard_page():
    """Render leaderboard page"""
    st.subheader("🏆 Diamond Hands Leaderboard")

    data_manager = get_data_manager()
    state = get_state()

    # Load leaderboard data
    df = data_manager.load_leaderboard()

    if df.empty:
        st.warning("⏳ No leaderboard data available yet.")
        st.info("💡 The analyzer needs at least 5 closed positions per KOL to generate rankings.")
        return

    # Search bar
    search_query = render_search_bar(
        placeholder="🔍 Search KOLs by name or wallet...",
        key="leaderboard_search"
    )

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        min_score = st.slider(
            "Min Diamond Hand Score",
            0, 100,
            int(state.get('filters.min_diamond_score', 0)),
            key="filter_score"
        )

    with col2:
        min_trades = st.slider(
            "Min Trades",
            1, 100,
            int(state.get('filters.min_trades', 1)),
            key="filter_trades"
        )

    with col3:
        show_only = st.selectbox(
            "Show",
            ["All", "Diamond Hands Only", "Scalpers Only"],
            key="filter_type"
        )

    # Apply filters
    filtered = df[df['diamond_hand_score'] >= min_score].copy()
    filtered = filtered[filtered['total_trades'] >= min_trades]

    if show_only == "Diamond Hands Only":
        filtered = filtered[filtered['is_diamond_hand'] == True]
    elif show_only == "Scalpers Only":
        filtered = filtered[filtered['is_scalper'] == True]

    # Apply search
    if search_query:
        from dashboard.components.search import FuzzySearch
        searcher = FuzzySearch(score_cutoff=70)
        filtered = searcher.search_kols(search_query, filtered, limit=1000)

    # Display
    if filtered.empty:
        st.warning("😕 No KOLs match your filters.")
    else:
        # Format display columns
        display_df = filtered[[
            'rank', 'name', 'diamond_hand_score', 'total_trades',
            'three_x_plus_rate', 'win_rate', 'avg_hold_time_hours', 'total_pnl_sol'
        ]].head(100).copy()

        display_df['three_x_plus_rate'] = (display_df['three_x_plus_rate'] * 100).round(1).astype(str) + '%'
        display_df['win_rate'] = (display_df['win_rate'] * 100).round(1).astype(str) + '%'
        display_df['avg_hold_time_hours'] = display_df['avg_hold_time_hours'].round(2)
        display_df['total_pnl_sol'] = display_df['total_pnl_sol'].round(2)

        display_df.columns = [
            'Rank', 'KOL', 'Score', 'Trades',
            '3x+ Rate', 'Win Rate', 'Avg Hold (h)', 'Total PnL (SOL)'
        ]

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        st.caption(f"Showing {len(display_df)} of {len(filtered)} KOLs")


def render_recent_trades():
    """Render recent trades section"""
    st.subheader("🔄 Recent Trades")

    data_manager = get_data_manager()

    trades = data_manager.get_recent_trades(limit=20)

    if not trades:
        st.warning("⏳ No recent trades available")
        return

    # Format trades for display
    import pandas as pd

    trade_data = []
    for trade in trades:
        trade_data.append({
            'KOL': trade['kol_name'],
            'Operation': trade['operation'].upper(),
            'Token': trade['token_address'][:8] + '...',
            'Amount SOL': f"{trade['amount_sol']:.4f}",
            'Amount Token': f"{trade['amount_token']:.2f}",
            'DEX': trade['dex'] or 'Unknown',
            'Time': trade['timestamp'].strftime('%Y-%m-%d %H:%M')
        })

    df = pd.DataFrame(trade_data)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_search_page():
    """Render fuzzy search page"""
    st.subheader("🔍 Advanced KOL Search")

    data_manager = get_data_manager()
    df = data_manager.load_leaderboard()

    if df.empty:
        st.warning("⏳ No leaderboard data available for search")
        return

    # Use fuzzy search component
    selected = fuzzy_search_component(
        data_source="kols",
        data_df=df,
        on_select=lambda row: display_kol_details(row['kol_id'])
    )

    if selected:
        display_kol_details(selected)


def display_kol_details(kol_id: int):
    """Display detailed KOL information"""
    data_manager = get_data_manager()

    kol = data_manager.get_kol_details(kol_id)
    positions = data_manager.get_kol_positions(kol_id, limit=20)

    if not kol:
        st.error(f"KOL with ID {kol_id} not found")
        return

    st.markdown(f"### {kol['name']}")
    st.code(kol['wallet_address'], language="text")

    if positions:
        st.markdown("#### Recent Positions")
        import pandas as pd
        pos_df = pd.DataFrame(positions)
        st.dataframe(pos_df, use_container_width=True, hide_index=True)
    else:
        st.info("No closed positions available")


def main():
    """Main dashboard application"""
    # Initialize state FIRST (before using it)
    state = get_state()
    data_manager = get_data_manager()

    # Apply theme after state is initialized
    apply_theme_from_state()

    # Update state
    state.set('current_page', 'leaderboard')

    # Render components
    render_header()
    render_metrics()
    render_sidebar()

    # Main content
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "🏆 Leaderboard",
        "📊 Analytics",
        "🪙 Token Analysis",
        "⚖️ Comparisons",
        "🤖 ML Predictions",
        "🔍 KOL Search",
        "🔄 Recent Trades",
        "📈 Summaries",
        "👤 KOL Details"
    ])

    with tab1:
        # Import and render enhanced leaderboard
        from dashboard.pages.leaderboard import main as leaderboard_main
        leaderboard_main()

    with tab2:
        # Import and render analytics
        from dashboard.pages.analytics import main as analytics_main
        analytics_main()

    with tab3:
        # Import and render token analysis
        from dashboard.pages.tokens import main as tokens_main
        tokens_main()

    with tab4:
        # Import and render comparisons
        from dashboard.pages.comparisons import main as comparisons_main
        comparisons_main()

    with tab5:
        # Import and render ML predictions
        from dashboard.pages.predictions import main as predictions_main
        predictions_main()

    with tab6:
        # Fuzzy search page
        render_search_page()

    with tab7:
        render_recent_trades()

    with tab8:
        # Summaries page
        from dashboard.pages.summaries import main as summaries_main
        summaries_main()

    with tab9:
        # KOL Details page
        from dashboard.pages.kol_details import main as kol_details_main
        kol_details_main()

    # Footer with enhanced info
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: var(--text-secondary); padding: 2rem 0;'>
            <p style='font-size: 1.1rem; margin-bottom: 0.5rem;'>
                <strong>KOL Tracker ML v2.1</strong> - Enhanced Modular Dashboard
            </p>
            <p style='font-size: 0.9rem;'>
                ✨ Fuzzy Search | 🤖 ML Predictions | 📊 Advanced Analytics
            </p>
            <p style='font-size: 0.85rem; margin-top: 0.5rem;'>
                Made with ❤️ for Solana traders | Data refreshes automatically
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Auto-refresh if enabled (with safe rerun)
    if state.get_preference('auto_refresh', False):
        import time

        # Show countdown
        interval = state.get_preference('refresh_interval', 30)

        # Progress bar for refresh
        progress_bar = st.progress(0)
        for i in range(interval):
            time.sleep(1)
            progress_bar.progress((i + 1) / interval)

        st.rerun()


if __name__ == "__main__":
    main()
