"""
Comparisons & Benchmarks Page

KOL comparison tools with:
- KOL vs KOL comparison (radar chart)
- Benchmark vs market
- Percentile rankings
- Performance attribution
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard.core.state_manager import get_state
from dashboard.core.data_manager import get_data_manager


def render_kol_vs_kol_comparison():
    """Render KOL vs KOL comparison radar chart"""
    st.markdown("### 👥 KOL vs KOL Comparison")

    data_manager = get_data_manager()
    df = data_manager.load_leaderboard()

    if df.empty or len(df) < 2:
        st.warning("⏳ Need at least 2 KOLs for comparison")
        return

    # KOL selectors
    kol_names = df['name'].tolist()

    col1, col2 = st.columns(2)

    with col1:
        kol1_name = st.selectbox("Select First KOL", kol_names, key="kol1_select")

    with col2:
        kol2_name = st.selectbox("Select Second KOL", kol_names, index=min(1, len(kol_names)-1), key="kol2_select")

    if not kol1_name or not kol2_name:
        return

    # Get KOL data
    kol1_data = df[df['name'] == kol1_name].iloc[0]
    kol2_data = df[df['name'] == kol2_name].iloc[0]

    # Metrics for comparison
    metrics = [
        'diamond_hand_score',
        'three_x_plus_rate',
        'win_rate',
        'consistency_score'
    ]

    # Normalize to 0-100
    kol1_values = []
    kol2_values = []

    for metric in metrics:
        if metric == 'diamond_hand_score':
            val1 = kol1_data[metric]
            val2 = kol2_data[metric]
        elif metric in ['three_x_plus_rate', 'win_rate']:
            val1 = kol1_data[metric] * 100
            val2 = kol2_data[metric] * 100
        elif metric == 'consistency_score':
            val1 = kol1_data.get(metric, 0) * 100
            val2 = kol2_data.get(metric, 0) * 100
        else:
            val1 = 0
            val2 = 0

        kol1_values.append(val1)
        kol2_values.append(val2)

    # Create radar chart
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=kol1_values + [kol1_values[0]],
        theta=['DH Score', '3x+ Rate', 'Win Rate', 'Consistency', 'DH Score'],
        fill='toself',
        name=kol1_name,
        line_color='#4ECDC4'
    ))

    fig.add_trace(go.Scatterpolar(
        r=kol2_values + [kol2_values[0]],
        theta=['DH Score', '3x+ Rate', 'Win Rate', 'Consistency', 'DH Score'],
        fill='toself',
        name=kol2_name,
        line_color='#FF6B6B'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=True,
        title="KOL Comparison Radar Chart",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    # Side-by-side comparison table
    st.markdown("#### 📊 Detailed Comparison")

    comparison_data = {
        'Metric': ['Diamond Hand Score', '3x+ Rate', 'Win Rate', 'Avg Hold (h)', 'Total Trades', 'Total PnL (SOL)'],
        kol1_name: [
            f"{kol1_data['diamond_hand_score']:.1f}",
            f"{kol1_data['three_x_plus_rate']*100:.1f}%",
            f"{kol1_data['win_rate']*100:.1f}%",
            f"{kol1_data['avg_hold_time_hours']:.2f}",
            int(kol1_data['total_trades']),
            f"{kol1_data['total_pnl_sol']:.2f}"
        ],
        kol2_name: [
            f"{kol2_data['diamond_hand_score']:.1f}",
            f"{kol2_data['three_x_plus_rate']*100:.1f}%",
            f"{kol2_data['win_rate']*100:.1f}%",
            f"{kol2_data['avg_hold_time_hours']:.2f}",
            int(kol2_data['total_trades']),
            f"{kol2_data['total_pnl_sol']:.2f}"
        ]
    }

    comp_df = pd.DataFrame(comparison_data)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)


def render_benchmark_comparison():
    """Render benchmark vs market comparison"""
    st.markdown("### 📊 Benchmark Comparisons")

    data_manager = get_data_manager()
    df = data_manager.load_leaderboard()

    if df.empty:
        st.warning("⏳ No leaderboard data available")
        return

    # Select KOL to benchmark
    kol_name = st.selectbox("Select KOL to Benchmark", df['name'].tolist(), key="benchmark_kol")

    if not kol_name:
        return

    kol_data = df[df['name'] == kol_name].iloc[0]

    # Calculate benchmarks
    avg_dh_score = df['diamond_hand_score'].mean()
    avg_win_rate = df['win_rate'].mean()
    avg_3x_rate = df['three_x_plus_rate'].mean()

    top10_dh_score = df.nlargest(10, 'diamond_hand_score')['diamond_hand_score'].mean()
    top10_win_rate = df.nlargest(10, 'diamond_hand_score')['win_rate'].mean()
    top10_3x_rate = df.nlargest(10, 'diamond_hand_score')['three_x_plus_rate'].mean()

    # Create comparison chart
    benchmarks = ['Market Avg', 'Top 10 Avg', 'Selected KOL']
    dh_scores = [avg_dh_score, top10_dh_score, kol_data['diamond_hand_score']]
    win_rates = [avg_win_rate * 100, top10_win_rate * 100, kol_data['win_rate'] * 100]
    three_x_rates = [avg_3x_rate * 100, top10_3x_rate * 100, kol_data['three_x_plus_rate'] * 100]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Diamond Hand Score', 'Win Rate (%)', '3x+ Rate (%)')
    )

    fig.add_trace(go.Bar(
        x=benchmarks,
        y=dh_scores,
        name='DH Score',
        marker_color=['#95E1D3', '#FFD700', '#4ECDC4'],
        showlegend=False
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=benchmarks,
        y=win_rates,
        name='Win Rate',
        marker_color=['#95E1D3', '#FFD700', '#4ECDC4'],
        showlegend=False
    ), row=1, col=2)

    fig.add_trace(go.Bar(
        x=benchmarks,
        y=three_x_rates,
        name='3x+ Rate',
        marker_color=['#95E1D3', '#FFD700', '#4ECDC4'],
        showlegend=False
    ), row=1, col=3)

    fig.update_layout(height=400, showlegend=False)

    st.plotly_chart(fig, use_container_width=True)

    # Percentile calculation
    dh_percentile = (df['diamond_hand_score'] < kol_data['diamond_hand_score']).sum() / len(df) * 100
    win_percentile = (df['win_rate'] < kol_data['win_rate']).sum() / len(df) * 100

    col1, col2 = st.columns(2)

    with col1:
        st.metric(f"{kol_name} DH Percentile", f"{dh_percentile:.1f}%")

    with col2:
        st.metric(f"{kol_name} Win Rate Percentile", f"{win_percentile:.1f}%")


def render_percentile_rankings():
    """Render percentile rankings for all metrics"""
    st.markdown("### 📈 Percentile Rankings")

    data_manager = get_data_manager()
    df = data_manager.load_leaderboard()

    if df.empty:
        st.warning("⏳ No leaderboard data available")
        return

    # Select KOL
    kol_name = st.selectbox("Select KOL", df['name'].tolist(), key="percentile_kol")

    if not kol_name:
        return

    kol_data = df[df['name'] == kol_name].iloc[0]

    # Calculate percentiles
    metrics = [
        ('Diamond Hand Score', kol_data['diamond_hand_score'], df['diamond_hand_score']),
        ('3x+ Rate', kol_data['three_x_plus_rate'] * 100, df['three_x_plus_rate'] * 100),
        ('Win Rate', kol_data['win_rate'] * 100, df['win_rate'] * 100),
        ('Total Trades', kol_data['total_trades'], df['total_trades']),
        ('Consistency', kol_data.get('consistency_score', 0) * 100, df.get('consistency_score', pd.Series([0]*len(df))) * 100)
    ]

    percentiles = []
    metric_names = []

    for name, value, all_values in metrics:
        percentile = (all_values < value).sum() / len(all_values) * 100
        percentiles.append(percentile)
        metric_names.append(name)

    # Create bar chart
    colors = ['#FF6B6B' if p < 25 else '#FFD700' if p < 50 else '#FFD700' if p < 75 else '#4ECDC4'
                for p in percentiles]

    fig = go.Figure(data=[
        go.Bar(
            x=metric_names,
            y=percentiles,
            marker_color=colors,
            text=[f"{p:.1f}%" for p in percentiles],
            textposition='outside',
        )
    ])

    fig.update_layout(
        title=f"Percentile Rankings: {kol_name}",
        xaxis_title="Metric",
        yaxis_title="Percentile (%)",
        yaxis=dict(range=[0, 100]),
        height=500,
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    # Show explanation
    st.info("📊 Percentile shows what percentage of KOLs score below the selected KOL. "
            "Higher is better - 90th percentile means the KOL scores better than 90% of all KOLs.")


def render_performance_attribution():
    """Render performance attribution analysis"""
    st.markdown("### 🎯 Performance Attribution")

    data_manager = get_data_manager()
    df = data_manager.load_leaderboard()

    if df.empty:
        st.warning("⏳ No leaderboard data available")
        return

    # Select KOL
    kol_name = st.selectbox("Select KOL for Analysis", df['name'].tolist(), key="attribution_kol")

    if not kol_name:
        return

    kol_data = df[df['name'] == kol_name].iloc[0]

    # Get KOL positions
    positions = data_manager.get_kol_positions(kol_data['kol_id'], limit=100)

    if not positions:
        st.warning("⏳ No position data available for this KOL")
        return

    pos_df = pd.DataFrame(positions)

    # Analyze performance contributors
    st.markdown(f"#### 📊 Performance Breakdown: {kol_name}")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Positions", len(pos_df))

    with col2:
        profitable = pos_df[pos_df['is_profitable'] == True]
        st.metric("Win Rate", f"{len(profitable)/len(pos_df)*100:.1f}%")

    with col3:
        three_x = pos_df[pos_df['pnl_multiple'] >= 3.0]
        st.metric("3x+ Count", len(three_x))

    with col4:
        avg_multiple = pos_df['pnl_multiple'].mean()
        st.metric("Avg Multiple", f"{avg_multiple:.2f}x")

    # Performance by hold time
    st.markdown("##### ⏰ Performance by Hold Time")

    pos_df['hold_category'] = pd.cut(
        pos_df['hold_time_hours'],
        bins=[0, 1, 6, 24, float('inf')],
        labels=['< 1h', '1-6h', '6-24h', '> 24h']
    )

    hold_performance = pos_df.groupby('hold_category').agg({
        'pnl_multiple': 'mean',
        'is_profitable': 'mean',
        'pnl_sol': 'sum'
    }).reset_index()

    hold_performance['is_profitable'] = (hold_performance['is_profitable'] * 100).round(1)

    st.dataframe(
        hold_performance.rename(columns={
            'hold_category': 'Hold Time',
            'pnl_multiple': 'Avg Multiple (x)',
            'is_profitable': 'Win Rate (%)',
            'pnl_sol': 'Total PnL (SOL)'
        }),
        use_container_width=True,
        hide_index=True
    )


def render_custom_benchmark():
    """Render custom benchmark creator"""
    st.markdown("### ⚙️ Create Custom Benchmark")

    data_manager = get_data_manager()
    df = data_manager.load_leaderboard()

    if df.empty:
        st.warning("⏳ No leaderboard data available")
        return

    st.markdown("Select KOLs to include in your custom benchmark:")

    # Multi-select KOLs
    selected_kols = st.multiselect(
        "Select KOLs",
        df['name'].tolist(),
        default=[],
        max_selections=20,
        key="custom_benchmark_kols"
    )

    if len(selected_kols) < 2:
        st.info("👈 Select at least 2 KOLs to create a benchmark")
        return

    # Get selected KOLs data
    benchmark_df = df[df['name'].isin(selected_kols)]

    # Calculate benchmark metrics
    benchmark_metrics = {
        'KOLs': len(selected_kols),
        'Avg DH Score': benchmark_df['diamond_hand_score'].mean(),
        'Avg Win Rate': benchmark_df['win_rate'].mean() * 100,
        'Avg 3x+ Rate': benchmark_df['three_x_plus_rate'].mean() * 100,
        'Total Trades': benchmark_df['total_trades'].sum(),
        'Total PnL': benchmark_df['total_pnl_sol'].sum()
    }

    # Display benchmark stats
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("KOLs in Benchmark", benchmark_metrics['KOLs'])

    with col2:
        st.metric("Avg DH Score", f"{benchmark_metrics['Avg DH Score']:.1f}")

    with col3:
        st.metric("Avg Win Rate", f"{benchmark_metrics['Avg Win Rate']:.1f}%")

    # Show KOL list
    st.markdown("#### 📋 Benchmark Composition")

    display_df = benchmark_df[[
        'name', 'diamond_hand_score', 'total_trades',
        'win_rate', 'total_pnl_sol'
    ]].copy()

    display_df['win_rate'] = (display_df['win_rate'] * 100).round(1).astype(str) + '%'
    display_df.columns = ['Name', 'DH Score', 'Trades', 'Win Rate', 'PnL (SOL)']

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Save benchmark button
    if st.button("💾 Save as Preset"):
        state = get_state()
        state.save_preset(
            name=f"Custom Benchmark ({len(selected_kols)} KOLs)",
            description=f"Benchmark of {len(selected_kols)} KOLs"
        )
        st.success("✅ Benchmark saved as preset!")


def main():
    """Main entry point for comparisons page"""
    st.title("⚖️ Comparisons & Benchmarks")

    # Tab navigation
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "👥 KOL vs KOL",
        "📊 Benchmarks",
        "📈 Percentiles",
        "🎯 Attribution",
        "⚙️ Custom Benchmark"
    ])

    with tab1:
        render_kol_vs_kol_comparison()

    with tab2:
        render_benchmark_comparison()

    with tab3:
        render_percentile_rankings()

    with tab4:
        render_performance_attribution()

    with tab5:
        render_custom_benchmark()


if __name__ == "__main__":
    main()
