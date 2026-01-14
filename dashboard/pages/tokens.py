"""
Token Analysis Page

Advanced token analytics with:
- Top tokens by KOLs
- Token performance matrix
- Entry timing analysis
- Success rate by token
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard.core.state_manager import get_state
from dashboard.core.data_manager import get_data_manager


def render_top_tokens():
    """Render top tokens by number of KOLs"""
    st.markdown("### 🏆 Top Tokens by KOL Count")

    data_manager = get_data_manager()

    # Get token analysis data
    token_df = data_manager.get_token_analysis()

    if token_df.empty:
        st.warning("⏳ No token data available yet")
        return

    # Sort by unique KOLs
    top_tokens = token_df.nlargest(20, 'unique_kols')

    # Create bar chart
    fig = go.Figure()

    # Color by avg multiple
    colors = ['#FF6B6B' if x < 1 else '#FFD700' if x < 3 else '#4ECDC4'
               for x in top_tokens['avg_multiple']]

    fig.add_trace(go.Bar(
        x=top_tokens['unique_kols'],
        y=top_tokens['token_address'].str[:8] + '...',
        orientation='h',
        marker_color=colors,
        text=[f"{x:.1f}x" for x in top_tokens['avg_multiple']],
        textposition='outside',
        name='Tokens'
    ))

    fig.update_layout(
        title="Top 20 Tokens by Number of KOLs",
        xaxis_title="Number of KOLs",
        yaxis_title="Token",
        height=600,
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    # Show data table
    display_df = top_tokens[[
        'token_address', 'unique_kols', 'trade_count',
        'avg_multiple', 'total_pnl', 'three_x_count'
    ]].copy()

    display_df['token_address'] = display_df['token_address'].str[:12] + '...'
    display_df['avg_multiple'] = display_df['avg_multiple'].round(2)
    display_df['total_pnl'] = display_df['total_pnl'].round(2)

    display_df.columns = [
        'Token', 'KOLs', 'Trades',
        'Avg Multiple', 'Total PnL', '3x+ Count'
    ]

    st.dataframe(display_df, use_container_width=True, hide_index=True)


def render_token_performance_matrix():
    """Render token performance heatmap"""
    st.markdown("### 🟦 Token Performance Matrix")

    data_manager = get_data_manager()
    df_leaderboard = data_manager.load_leaderboard()

    if df_leaderboard.empty:
        st.warning("⏳ No leaderboard data available")
        return

    # Get top tokens
    token_df = data_manager.get_token_analysis()

    if token_df.empty:
        st.warning("⏳ No token data available")
        return

    # Get top 20 tokens by unique KOLs
    top_tokens = token_df.nlargest(20, 'unique_kols')['token_address'].tolist()

    # Get KOL-performance data for these tokens
    from core.database import db, ClosedPosition
    from sqlalchemy import func

    session = db.get_session()

    try:
        # Query: KOL vs Token performance
        query = session.query(
            ClosedPosition.kol_id,
            ClosedPosition.token_address,
            func.avg(ClosedPosition.pnl_multiple).label('avg_multiple'),
            func.count(ClosedPosition.id).label('trade_count')
        ).filter(
            ClosedPosition.token_address.in_(top_tokens)
        ).group_by(
            ClosedPosition.kol_id,
            ClosedPosition.token_address
        )

        df = pd.read_sql(query.statement, session.bind)

        if df.empty:
            st.warning("⏳ No KOL-token interaction data available")
            return

        # Get top KOLs
        top_kols = df_leaderboard.nlargest(15, 'diamond_hand_score')['kol_id'].tolist()

        # Filter for top KOLs
        df = df[df['kol_id'].isin(top_kols)]

        # Create pivot table (KOLs vs Tokens)
        pivot_df = df.pivot_table(
            index='kol_id',
            columns='token_address',
            values='avg_multiple',
            fill_value=0
        )

        # Get KOL names
        kol_names = dict(zip(
            df_leaderboard['kol_id'],
            df_leaderboard['name']
        ))

        # Format labels
        token_labels = [t[:8] + '...' for t in pivot_df.columns]
        kol_labels = [kol_names.get(k, f"KOL {k}") for k in pivot_df.index]

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot_df.values,
            x=token_labels,
            y=kol_labels,
            colorscale=[
                [0, '#FF6B6B'],    # Red (<1x)
                [0.33, '#FFD700'],  # Yellow (1-3x)
                [0.67, '#4ECDC4'],  # Cyan (>3x)
                [1, '#95E1D3']      # Light cyan
            ],
            colorbar=dict(title="Avg Multiple (x)"),
            text=np.round(pivot_df.values, 1),
            texttemplate='%{text}',
            textfont={"size": 9}
        ))

        fig.update_layout(
            title="Token Performance by Top KOLs",
            xaxis_title="Token",
            yaxis_title="KOL",
            height=600
        )

        st.plotly_chart(fig, use_container_width=True)

    finally:
        session.close()


def render_token_timing_analysis():
    """Render token entry timing analysis"""
    st.markdown("### ⏰ Token Entry Timing")

    data_manager = get_data_manager()
    df_leaderboard = data_manager.load_leaderboard()

    if df_leaderboard.empty:
        st.warning("⏳ No leaderboard data available")
        return

    # Get closed positions
    from core.database import db, ClosedPosition
    from sqlalchemy import extract

    session = db.get_session()

    try:
        # Query positions with hour of entry
        query = session.query(
            extract('hour', ClosedPosition.entry_time).label('hour'),
            ClosedPosition.pnl_multiple,
            ClosedPosition.pnl_sol
        ).filter(
            ClosedPosition.entry_time >= pd.Timestamp.now() - pd.Timedelta(days=30)
        )

        df = pd.read_sql(query.statement, session.bind)

        if df.empty:
            st.warning("⏳ Not enough data for timing analysis")
            return

        # Create scatter plot - use absolute values for size to avoid negative values
        if 'pnl_sol' in df.columns:
            df['abs_pnl_sol'] = df['pnl_sol'].abs()
            size_column = 'abs_pnl_sol'
        else:
            size_column = None

        fig = px.scatter(
            df,
            x='hour',
            y='pnl_multiple',
            color='pnl_multiple',
            size=size_column,
            title="Entry Hour vs Multiple (Last 30 Days)",
            labels={
                'hour': 'Entry Hour (0-23)',
                'pnl_multiple': 'Multiple (x)',
                'abs_pnl_sol': '|PnL| (SOL)'
            },
            color_continuous_scale='RdYlGn',
            range_color=[0, 5],
            height=500
        )

        # Add 3x reference line
        fig.add_hline(y=3.0, line_dash="dash", line_color="white",
                     annotation_text="3x Target")

        fig.update_layout(
            xaxis=dict(tickmode='linear', tick0=0, dtick=2),
            yaxis_type="log"
        )

        st.plotly_chart(fig, use_container_width=True)

    finally:
        session.close()


def render_success_rate_by_token():
    """Render success rate analysis by token"""
    st.markdown("### 📈 Success Rate by Token")

    data_manager = get_data_manager()
    token_df = data_manager.get_token_analysis()

    if token_df.empty:
        st.warning("⏳ No token data available")
        return

    # Calculate success rate
    token_df['success_rate'] = (
        token_df['three_x_count'] / token_df['trade_count']
    ).fillna(0)

    # Get top tokens by trade count
    top_tokens = token_df.nlargest(30, 'trade_count')

    # Create bar chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=top_tokens['token_address'].str[:8] + '...',
        y=top_tokens['success_rate'] * 100,
        marker_color=top_tokens['success_rate'] * 100,
        text=[f"{x:.1f}%" for x in top_tokens['success_rate'] * 100],
        textposition='outside',
        name='Success Rate'
    ))

    # Color scale
    fig.update_traces(
        marker_colorscale='RdYlGn',
        marker_showscale=True,
        marker_colorbar_title="Success Rate (%)"
    )

    fig.update_layout(
        title="Token Success Rate (3x+ Rate by Token)",
        xaxis_title="Token",
        yaxis_title="Success Rate (%)",
        height=500,
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)


def render_token_explorer():
    """Render interactive token explorer"""
    st.markdown("### 🔍 Token Explorer")

    data_manager = get_data_manager()
    token_df = data_manager.get_token_analysis()

    if token_df.empty:
        st.warning("⏳ No token data available")
        return

    # Token selector
    token_options = token_df['token_address'].str[:12] + '... (' + \
                    token_df['trade_count'].astype(str) + ' trades)'
    selected_token_idx = st.selectbox(
        "Select Token to Explore",
        range(len(token_options)),
        format_func=lambda i: token_options.iloc[i]
    )

    if selected_token_idx is None:
        return

    selected_token = token_df.iloc[selected_token_idx]['token_address']

    # Get KOLs who traded this token
    from core.database import db, ClosedPosition, KOL

    session = db.get_session()

    try:
        positions = session.query(
            ClosedPosition,
            KOL
        ).join(
            KOL, ClosedPosition.kol_id == KOL.id
        ).filter(
            ClosedPosition.token_address == selected_token
        ).order_by(
            ClosedPosition.exit_time.desc()
        ).limit(50).all()

        if not positions:
            st.info("No recent positions found for this token")
            return

        # Display token stats
        col1, col2, col3, col4 = st.columns(4)

        token_stats = token_df[token_df['token_address'] == selected_token].iloc[0]

        with col1:
            st.metric("Total Trades", int(token_stats['trade_count']))

        with col2:
            st.metric("Unique KOLs", int(token_stats['unique_kols']))

        with col3:
            st.metric("Avg Multiple", f"{token_stats['avg_multiple']:.2f}x")

        with col4:
            st.metric("3x+ Count", int(token_stats['three_x_count']))

        # Display positions
        st.markdown("#### Recent Positions")

        position_data = []
        for pos, kol in positions:
            position_data.append({
                'KOL': kol.name,
                'Entry': pos.entry_time.strftime('%Y-%m-%d %H:%M'),
                'Exit': pos.exit_time.strftime('%Y-%m-%d %H:%M'),
                'Hold (h)': f"{pos.hold_time_hours:.1f}",
                'Multiple': f"{pos.pnl_multiple:.2f}x",
                'PnL (SOL)': f"{pos.pnl_sol:.2f}"
            })

        pos_df = pd.DataFrame(position_data)
        st.dataframe(pos_df, use_container_width=True, hide_index=True)

    finally:
        session.close()


def main():
    """Main entry point for token analysis page"""
    st.title("🪙 Token Analysis")

    # Tab navigation
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏆 Top Tokens",
        "🟦 Performance Matrix",
        "⏰ Entry Timing",
        "📈 Success Rate",
        "🔍 Token Explorer"
    ])

    with tab1:
        render_top_tokens()

    with tab2:
        render_token_performance_matrix()

    with tab3:
        render_token_timing_analysis()

    with tab4:
        render_success_rate_by_token()

    with tab5:
        render_token_explorer()


if __name__ == "__main__":
    main()
