"""
Analytics Page

Advanced temporal analytics with:
- Time series trends
- Activity heatmaps
- Distribution analysis
- Correlation matrices
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

from dashboard.core.state_manager import get_state
from dashboard.core.data_manager import get_data_manager


def render_time_series_trends():
    """Render time series trend analysis"""
    st.markdown("### 📈 Performance Trends Over Time")

    data_manager = get_data_manager()

    # Granularity selector
    col1, col2 = st.columns(2)

    with col1:
        granularity = st.selectbox(
            "Time Granularity",
            ['day', 'week', 'month'],
            index=0,
            key="analytics_granularity"
        )

    with col2:
        days_back = st.selectbox(
            "Lookback Period",
            [7, 14, 30, 60, 90, 180],
            index=4,
            key="analytics_days_back"
        )

    # Get time series data
    df = data_manager.get_time_series_data(granularity=granularity, days_back=days_back)

    if df.empty:
        st.warning("⏳ Not enough historical data for time series analysis")
        return

    # Create date column
    if granularity == 'month':
        df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
    else:
        df['date'] = pd.to_datetime(df[['year', 'month', 'day']])

    # Sort by date
    df = df.sort_values('date')

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Trades Over Time',
            'Avg Multiple Over Time',
            'Total PnL Over Time',
            'Trade Count (Bar)'
        ),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"type": "bar"}]
        ]
    )

    # Plot 1: Trades over time
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['trade_count'],
            mode='lines+markers',
            name='Trades',
            line=dict(color='#4ECDC4', width=2)
        ),
        row=1, col=1
    )

    # Plot 2: Avg multiple over time
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['avg_multiple'],
            mode='lines+markers',
            name='Avg Multiple',
            line=dict(color='#FF6B6B', width=2)
        ),
        row=1, col=2
    )

    # Add 3x reference line
    fig.add_hline(
        y=3.0,
        line_dash="dash",
        line_color="gray",
        annotation_text="3x Target",
        row=1, col=2
    )

    # Plot 3: Total PnL over time
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['total_pnl'],
            mode='lines+markers',
            name='Total PnL (SOL)',
            line=dict(color='#95E1D3', width=2),
            fill='tozeroy'
        ),
        row=2, col=1
    )

    # Plot 4: Bar chart
    fig.add_trace(
        go.Bar(
            x=df['date'],
            y=df['trade_count'],
            name='Trade Count',
            marker_color='#4ECDC4'
        ),
        row=2, col=2
    )

    # Update layout
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=2)
    fig.update_yaxes(title_text="Trades", row=1, col=1)
    fig.update_yaxes(title_text="Multiple (x)", row=1, col=2)
    fig.update_yaxes(title_text="PnL (SOL)", row=2, col=1)
    fig.update_yaxes(title_text="Trades", row=2, col=2)

    fig.update_layout(
        height=800,
        showlegend=False,
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)


def render_activity_heatmap():
    """Render activity heatmap (hour vs day of week)"""
    st.markdown("### 🌡️ Trading Activity Heatmap")

    data_manager = get_data_manager()
    session = data_manager.get_data_manager() if hasattr(data_manager, 'get_data_manager') else data_manager

    from core.database import db, Trade
    from sqlalchemy import func, extract

    session_db = db.get_session()

    try:
        # Get trades with hour and day of week
        query = session_db.query(
            extract('hour', Trade.timestamp).label('hour'),
            extract('dow', Trade.timestamp).label('dow'),
            func.count(Trade.id).label('trade_count')
        ).filter(
            Trade.timestamp >= datetime.now() - timedelta(days=30)
        ).group_by(
            extract('hour', Trade.timestamp),
            extract('dow', Trade.timestamp)
        )

        df = pd.read_sql(query.statement, session_db.bind)

        if df.empty:
            st.warning("⏳ Not enough data for heatmap")
            return

        # Pivot to create matrix
        heatmap_data = df.pivot(
            index='hour',
            columns='dow',
            values='trade_count'
        ).fillna(0)

        # Day of week labels
        dow_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=[dow_labels[int(i)] if i < 7 else str(i) for i in heatmap_data.columns],
            y=[f"{int(h)}:00" for h in heatmap_data.index],
            colorscale='Viridis',
            colorbar=dict(title="Trade Count")
        ))

        fig.update_layout(
            title="Trading Activity by Hour and Day of Week",
            xaxis_title="Day of Week",
            yaxis_title="Hour of Day",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

    finally:
        session_db.close()


def render_distribution_analysis():
    """Render distribution analysis of key metrics"""
    st.markdown("### 📊 Metric Distributions")

    data_manager = get_data_manager()
    df = data_manager.load_leaderboard()

    if df.empty:
        st.warning("⏳ No leaderboard data available")
        return

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Diamond Hand Score Distribution',
            'Win Rate Distribution',
            'Hold Time Distribution',
            '3x+ Rate Distribution'
        )
    )

    # Plot 1: DH Score
    fig.add_trace(
        go.Histogram(
            x=df['diamond_hand_score'],
            nbinsx=30,
            name='DH Score',
            marker_color='#4ECDC4'
        ),
        row=1, col=1
    )

    # Plot 2: Win Rate
    fig.add_trace(
        go.Histogram(
            x=df['win_rate'] * 100,
            nbinsx=30,
            name='Win Rate %',
            marker_color='#FF6B6B'
        ),
        row=1, col=2
    )

    # Plot 3: Hold Time (log scale)
    hold_times = df['avg_hold_time_hours'].clip(upper=100)
    fig.add_trace(
        go.Histogram(
            x=hold_times,
            nbinsx=30,
            name='Hold Hours',
            marker_color='#95E1D3'
        ),
        row=2, col=1
    )

    # Plot 4: 3x+ Rate
    three_x_rate = df['three_x_plus_rate'] * 100
    fig.add_trace(
        go.Histogram(
            x=three_x_rate,
            nbinsx=30,
            name='3x+ Rate %',
            marker_color='#DDA0DD'
        ),
        row=2, col=2
    )

    fig.update_layout(
        height=700,
        showlegend=False,
        title_text="Distribution of Key Metrics"
    )

    st.plotly_chart(fig, use_container_width=True)


def render_correlation_matrix():
    """Render correlation matrix of key metrics"""
    st.markdown("### 🔗 Metric Correlations")

    data_manager = get_data_manager()
    df = data_manager.load_leaderboard()

    if df.empty:
        st.warning("⏳ No leaderboard data available")
        return

    # Select numeric columns for correlation
    numeric_cols = [
        'diamond_hand_score',
        'total_trades',
        'three_x_plus_rate',
        'win_rate',
        'avg_hold_time_hours',
        'total_pnl_sol',
        'consistency_score'
    ]

    # Filter columns that exist
    available_cols = [col for col in numeric_cols if col in df.columns]

    if len(available_cols) < 2:
        st.warning("⏳ Not enough numeric columns for correlation")
        return

    corr_df = df[available_cols].corr()

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_df.values,
        x=[col.replace('_', ' ').title() for col in corr_df.columns],
        y=[col.replace('_', ' ').title() for col in corr_df.index],
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_df.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))

    fig.update_layout(
        title="Correlation Matrix of Key Metrics",
        height=600,
        width=800
    )

    st.plotly_chart(fig, use_container_width=True)


def render_top_performers():
    """Render top performers analysis"""
    st.markdown("### 🏆 Top Performers Analysis")

    data_manager = get_data_manager()
    df = data_manager.load_leaderboard()

    if df.empty:
        st.warning("⏳ No leaderboard data available")
        return

    # Top 10 by different metrics
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🥇 Top 10 by Diamond Hand Score")
        top_dh = df.nlargest(10, 'diamond_hand_score')[['name', 'diamond_hand_score', 'total_trades', 'win_rate']]
        top_dh['win_rate'] = (top_dh['win_rate'] * 100).round(1).astype(str) + '%'
        st.dataframe(top_dh, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("#### 💰 Top 10 by Total PnL")
        top_pnl = df.nlargest(10, 'total_pnl_sol')[['name', 'total_pnl_sol', 'diamond_hand_score', 'total_trades']]
        st.dataframe(top_pnl, use_container_width=True, hide_index=True)

    # Bottom performers
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### ⚠️ Lowest Diamond Hand Scores")
        bottom_dh = df.nsmallest(10, 'diamond_hand_score')[['name', 'diamond_hand_score', 'total_trades']]
        st.dataframe(bottom_dh, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("#### 📉 Lowest Win Rates")
        bottom_wr = df.nsmallest(10, 'win_rate')[['name', 'win_rate', 'total_trades']]
        bottom_wr['win_rate'] = (bottom_wr['win_rate'] * 100).round(1).astype(str) + '%'
        st.dataframe(bottom_wr, use_container_width=True, hide_index=True)


def main():
    """Main entry point for analytics page"""
    st.title("📊 Advanced Analytics")

    # Tab navigation within analytics
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Trends",
        "🌡️ Activity Heatmap",
        "📊 Distributions",
        "🔗 Correlations",
        "🏆 Top Performers"
    ])

    with tab1:
        render_time_series_trends()

    with tab2:
        render_activity_heatmap()

    with tab3:
        render_distribution_analysis()

    with tab4:
        render_correlation_matrix()

    with tab5:
        render_top_performers()


if __name__ == "__main__":
    main()
