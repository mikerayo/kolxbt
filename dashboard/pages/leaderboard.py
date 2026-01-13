"""
Enhanced Leaderboard Page

Improved leaderboard with:
- Column sorting
- CSV/Excel export
- Mini sparklines
- Visual badges
- Advanced filters
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import plotly.graph_objects as go
from io import BytesIO

from dashboard.core.state_manager import get_state
from dashboard.core.data_manager import get_data_manager


def render_badges(score: float, is_diamond_hand: bool, is_scalper: bool) -> str:
    """
    Render HTML badges for KOL classification

    Args:
        score: Diamond hand score
        is_diamond_hand: Is diamond hand
        is_scalper: Is scalper

    Returns:
        HTML string with badges
    """
    badges = []

    if is_diamond_hand:
        badges.append("💎 <span style='color: #4ECDC4; font-weight: bold;'>DH</span>")

    if is_scalper:
        badges.append("⚡ <span style='color: #FF6B6B; font-weight: bold;'>SCALPER</span>")

    if score >= 80:
        badges.append("🏆 <span style='color: #FFD700; font-weight: bold;'>ELITE</span>")
    elif score >= 50:
        badges.append("⭐ <span style='color: #C0C0C0; font-weight: bold;'>PRO</span>")

    return " | ".join(badges) if badges else ""


def render_sparkline(values: List[float], color: str = "#4ECDC4") -> go.Figure:
    """
    Create a mini sparkline chart

    Args:
        values: List of numeric values
        color: Line color

    Returns:
        Plotly figure
    """
    if not values or len(values) < 2:
        # Return empty figure
        fig = go.Figure()
        fig.update_layout(
            height=30,
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False
        )
        return fig

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(range(len(values))),
        y=values,
        mode='lines',
        line=dict(color=color, width=1.5),
        fill='tozeroy',
        fillcolor=f'{color}33'
    ))

    fig.update_layout(
        height=40,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False)
    )

    return fig


def export_to_csv(df: pd.DataFrame, filename: str = "leaderboard.csv") -> bytes:
    """
    Export DataFrame to CSV bytes

    Args:
        df: DataFrame to export
        filename: Suggested filename

    Returns:
        CSV bytes
    """
    output = BytesIO()
    df.to_csv(output, index=False, encoding='utf-8-sig')
    return output.getvalue()


def export_to_excel(df: pd.DataFrame, filename: str = "leaderboard.xlsx") -> bytes:
    """
    Export DataFrame to Excel bytes

    Args:
        df: DataFrame to export
        filename: Suggested filename

    Returns:
        Excel bytes
    """
    output = BytesIO()

    # Try using openpyxl or fallback
    try:
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Leaderboard')
    except ImportError:
        # Fallback to CSV
        return export_to_csv(df, filename.replace('.xlsx', '.csv'))

    return output.getvalue()


def render_sortable_table(
    df: pd.DataFrame,
    key: str = "leaderboard"
) -> None:
    """
    Render sortable table with export buttons

    Args:
        df: DataFrame to display
        key: Unique key for component
    """
    # Initialize sort state
    if f'{key}_sort_column' not in st.session_state:
        st.session_state[f'{key}_sort_column'] = 'diamond_hand_score'
        st.session_state[f'{key}_sort_ascending'] = False

    # Export buttons
    col1, col2, col3 = st.columns([1, 1, 4])

    with col1:
        if st.button("📥 CSV", key=f"{key}_csv"):
            csv_data = export_to_csv(df)
            st.download_button(
                label="Download",
                data=csv_data,
                file_name="kol_leaderboard.csv",
                mime="text/csv",
                key=f"{key}_download_csv"
            )

    with col2:
        if st.button("📊 Excel", key=f"{key}_excel"):
            excel_data = export_to_excel(df)
            st.download_button(
                label="Download",
                data=excel_data,
                file_name="kol_leaderboard.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"{key}_download_excel"
            )

    # Sort controls
    sort_column = st.selectbox(
        "Sort by",
        options=['rank', 'name', 'diamond_hand_score', 'total_trades',
                 'three_x_plus_rate', 'win_rate', 'avg_hold_time_hours', 'total_pnl_sol'],
        index=['rank', 'name', 'diamond_hand_score', 'total_trades',
               'three_x_plus_rate', 'win_rate', 'avg_hold_time_hours', 'total_pnl_sol'].index(
            st.session_state[f'{key}_sort_column']
        ),
        key=f"{key}_sort_column_select"
    )

    sort_ascending = st.checkbox(
        "Ascending",
        value=st.session_state[f'{key}_sort_ascending'],
        key=f"{key}_sort_asc_checkbox"
    )

    # Update sort state
    st.session_state[f'{key}_sort_column'] = sort_column
    st.session_state[f'{key}_sort_ascending'] = sort_ascending

    # Apply sort
    df_sorted = df.sort_values(by=sort_column, ascending=sort_ascending)

    # Format display DataFrame
    display_columns = [
        'rank', 'name', 'diamond_hand_score', 'total_trades',
        'three_x_plus_rate', 'win_rate', 'avg_hold_time_hours', 'total_pnl_sol'
    ]

    display_df = df_sorted[display_columns].copy()

    # Format columns
    display_df['three_x_plus_rate'] = (display_df['three_x_plus_rate'] * 100).round(1).astype(str) + '%'
    display_df['win_rate'] = (display_df['win_rate'] * 100).round(1).astype(str) + '%'
    display_df['avg_hold_time_hours'] = display_df['avg_hold_time_hours'].round(2)
    display_df['total_pnl_sol'] = display_df['total_pnl_sol'].round(2)

    # Rename columns
    display_df.columns = [
        'Rank', 'KOL', 'Score', 'Trades',
        '3x+ Rate', 'Win Rate', 'Avg Hold (h)', 'Total PnL (SOL)'
    ]

    # Add badges column
    badges_list = []
    for _, row in df_sorted.iterrows():
        badge = render_badges(
            row['diamond_hand_score'],
            row['is_diamond_hand'],
            row['is_scalper']
        )
        badges_list.append(badge)

    display_df.insert(3, 'Type', badges_list)

    # Display with column config - use HTML to avoid PyArrow issues
    # Convert to HTML table
    st.markdown(
        display_df.to_html(index=False, escape=False, classes='dataframe'),
        unsafe_allow_html=True
    )

    # Show stats
    st.caption(f"📊 Showing {len(display_df)} KOLs")


def render_enhanced_leaderboard():
    """Render the enhanced leaderboard page"""
    st.subheader("🏆 Diamond Hands Leaderboard (Enhanced)")

    data_manager = get_data_manager()
    state = get_state()

    # Load leaderboard data
    df = data_manager.load_leaderboard()

    if df.empty:
        st.warning("⏳ No leaderboard data available yet.")
        st.info("💡 The analyzer needs at least 5 closed positions per KOL to generate rankings.")
        return

    # Search bar with fuzzy search
    from dashboard.components.search import render_search_bar, FuzzySearch

    search_query = render_search_bar(
        placeholder="🔍 Search KOLs by name or wallet...",
        key="leaderboard_search"
    )

    # Advanced filters expander
    with st.expander("🎛️ Advanced Filters", expanded=False):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            min_score = st.slider(
                "Min Diamond Hand Score",
                0, 100,
                int(state.get('filters.min_diamond_score', 0)),
                key="filter_score_enhanced"
            )

        with col2:
            min_trades = st.slider(
                "Min Trades",
                1, 100,
                int(state.get('filters.min_trades', 1)),
                key="filter_trades_enhanced"
            )

        with col3:
            max_trades = st.slider(
                "Max Trades",
                1, 500,
                int(state.get('filters.max_trades', 500)),
                key="filter_max_trades_enhanced"
            )

        with col4:
            show_only = st.selectbox(
                "KOL Type",
                ['all', 'diamond_hands_only', 'scalpers_only'],
                key="filter_type_enhanced"
            )

        # Additional filters
        col1, col2 = st.columns(2)

        with col1:
            win_rate_min = st.slider(
                "Min Win Rate (%)",
                0, 100,
                int(state.get('filters.win_rate_range', [0.0, 1.0])[0] * 100),
                key="filter_win_rate_min"
            )

        with col2:
            win_rate_max = st.slider(
                "Max Win Rate (%)",
                0, 100,
                int(state.get('filters.win_rate_range', [0.0, 1.0])[1] * 100),
                key="filter_win_rate_max"
            )

        # Preset buttons
        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("💎 Top Diamond Hands"):
                state.update_filters(
                    min_diamond_score=80,
                    min_trades=5,
                    kol_type='diamond_hands_only'
                )
                st.rerun()

        with col2:
            if st.button("⚡ High Scalpers"):
                state.update_filters(
                    min_trades=20,
                    kol_type='scalpers_only'
                )
                st.rerun()

        with col3:
            if st.button("🔄 Reset Filters"):
                state.clear_filters()
                st.rerun()

    # Apply filters
    filtered = df[df['diamond_hand_score'] >= min_score].copy()
    filtered = filtered[filtered['total_trades'] >= min_trades]
    filtered = filtered[filtered['total_trades'] <= max_trades]

    if show_only == "diamond_hands_only":
        filtered = filtered[filtered['is_diamond_hand'] == True]
    elif show_only == "scalpers_only":
        filtered = filtered[filtered['is_scalper'] == True]

    # Apply win rate filter
    filtered = filtered[
        (filtered['win_rate'] >= win_rate_min / 100) &
        (filtered['win_rate'] <= win_rate_max / 100)
    ]

    # Apply fuzzy search
    if search_query:
        searcher = FuzzySearch(score_cutoff=70)
        filtered = searcher.search_kols(search_query, filtered, limit=1000)

    # Display results count
    if not filtered.empty:
        st.info(f"📊 Found {len(filtered)} KOLs matching your criteria")
    else:
        st.warning("😕 No KOLs match your filters. Try adjusting the criteria.")
        return

    # Render sortable table with export
    render_sortable_table(filtered, key="enhanced_leaderboard")

    # Quick stats
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_score = filtered['diamond_hand_score'].mean()
        st.metric("Avg DH Score", f"{avg_score:.1f}")

    with col2:
        avg_win_rate = (filtered['win_rate'].mean() * 100)
        st.metric("Avg Win Rate", f"{avg_win_rate:.1f}%")

    with col3:
        total_trades = filtered['total_trades'].sum()
        st.metric("Total Trades", f"{int(total_trades):,}")

    with col4:
        diamond_hands = filtered['is_diamond_hand'].sum()
        st.metric("Diamond Hands", f"{int(diamond_hands)}")


def main():
    """Main entry point for leaderboard page"""
    render_enhanced_leaderboard()


if __name__ == "__main__":
    main()
