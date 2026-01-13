"""
Tooltips Component

Provides helpful tooltips and explanations for metrics and features.
"""

import streamlit as st


TOOLTIPS = {
    # Leaderboard metrics
    'diamond_hand_score': {
        'title': '💎 Diamond Hand Score',
        'content': '''
        **Overall KOL Quality Metric (0-100)**

        Combines multiple factors:
        - 🎯 3x+ Success Rate (40%)
        - 📈 Win Rate (30%)
        - ⏰ Average Hold Time (20%)
        - 🔥 Consistency (10%)

        **Higher is better** - Top KOLs have scores > 80
        '''
    },
    'three_x_plus_rate': {
        'title': '🎯 3x+ Rate',
        'content': '''
        **Percentage of positions that reached 3x or higher**

        Shows how often the KOL achieves multi-baggers.
        - 0%: Never hits 3x
        - 20%+: Excellent

        High rate = identifies moonshots early
        '''
    },
    'win_rate': {
        'title': '📈 Win Rate',
        'content': '''
        **Percentage of profitable positions**

        Consistency indicator:
        - < 40%: Poor
        - 40-60%: Average
        - 60-80%: Good
        - > 80%: Excellent

        High win rate = steady profits
        '''
    },
    'avg_hold_time_hours': {
        'title': '⏰ Average Hold Time',
        'content': '''
        **Average time from entry to exit (in hours)**

        Trading style indicator:
        - < 1h: Scalper (quick flips)
        - 1-6h: Day trader
        - 6-24h: Swing trader
        - > 24h: Diamond hand (long-term holder)

        Balance matters - too short = miss gains, too long = miss opportunities
        '''
    },
    'total_pnl_sol': {
        'title': '💰 Total PnL (SOL)',
        'content': '''
        **Total Profit/Loss in SOL**

        Sum of all closed positions:
- Positive = Profitable
- Negative = Loss

        **Note**: Doesn't include open positions
        '''
    },
    'total_trades': {
        'title': '📊 Total Trades',
        'content': '''
        **Total number of closed positions**

        Experience indicator:
        - < 10: New trader
        - 10-50: Active trader
        - 50-100: Experienced
        - > 100: Veteran

        More trades = more reliable stats
        '''
    },
    'consistency_score': {
        'title': '🔥 Consistency',
        'content': '''
        **How consistent is the KOL's performance?**

        Measures variance in returns:
        - High (> 80%): Stable, predictable performance
        - Medium (50-80%): Some volatility
        - Low (< 50%): Unpredictable, swings wildly

        Higher = more reliable
        '''
    },

    # ML Predictions
    'token_3x_probability': {
        'title': '🎯 Token 3x+ Probability',
        'content': '''
        **ML Prediction: Will this token reach 3x?**

        Based on:
        - KOL's historical performance
        - Amount invested
        - Entry time
        - Similar historical trades

        **Probability levels:**
        - 🔴 HIGH (> 70%): Strong signal
        - 🟡 MEDIUM (40-70%): Moderate signal
        - 🟢 LOW (< 40%): Weak signal

        ⚠️ Not financial advice - statistical prediction only
        '''
    },
    'model_auc': {
        'title': '📉 AUC-ROC',
        'content': '''
        **Area Under Curve - ROC Metric**

        Model quality measure:
        - 1.0: Perfect predictions
        - 0.9-0.99: Excellent
        - 0.8-0.89: Good
        - 0.7-0.79: Acceptable
        - 0.5: Random guessing
        - < 0.5: Worse than random

        Current model: 0.73 (Acceptable)
        '''
    },

    # Analytics metrics
    'sharpe_ratio': {
        'title': '📊 Sharpe Ratio',
        'content': '''
        **Risk-adjusted returns**

        Measures return per unit of risk:
        - < 1: Poor risk-adjusted performance
        - 1-2: Good
        - 2-3: Very good
        - > 3: Excellent

        Higher = better returns for the risk taken
        '''
    },
    'max_drawdown': {
        'title': '📉 Max Drawdown',
        'content': '''
        **Maximum peak-to-trough decline**

        Worst loss from a peak:
        - < 20%: Low risk
        - 20-40%: Moderate risk
        - > 40%: High risk

        Lower = less volatility, more stable
        '''
    },

    # Trading patterns
    'diamond_hand': {
        'title': '💎 Diamond Hand',
        'content': '''
        **Long-term holder with high 3x+ rate**

        Characteristics:
        - Hold time > 6 hours average
        - 3x+ rate > 15%
        - DH Score > 60

        These KOLs identify quality projects early and hold through volatility
        '''
    },
    'scalper': {
        'title': '⚡ Scalper',
        'content': '''
        **Quick flip trader**

        Characteristics:
        - Hold time < 1 hour average
        - High trade frequency
        - Win rate > 50%

        These KOLs profit from quick price movements
        '''
    },
    'sniper': {
        'title': '🎯 Sniper',
        'content': '''
        **Precision entry trader**

        Characteristics:
        - Win rate > 70%
        - 3x+ rate > 10%
        - Low trade count but high quality

        These KOLs are very selective with entries
        '''
    }
}


def render_tooltip(key, icon='ℹ️', label='What is this?'):
    """
    Render a tooltip with info icon

    Args:
        key: Tooltip key from TOOLTIPS dict
        icon: Icon to display (default: ℹ️)
        label: Label text (default: "What is this?")
    """
    if key not in TOOLTIPS:
        return

    tooltip = TOOLTIPS[key]

    col1, col2 = st.columns([1, 10])

    with col1:
        st.markdown(f"<sup>{icon}</sup>", unsafe_allow_html=True)

    with col2:
        with st.expander(label, expanded=False):
            st.markdown(f"##### {tooltip['title']}")
            st.markdown(tooltip['content'])
            st.markdown("---")


def render_metric_with_tooltip(label, value, tooltip_key, delta=None, help_text=None):
    """
    Render a Streamlit metric with an attached tooltip

    Args:
        label: Metric label
        value: Metric value
        tooltip_key: Key for tooltip content
        delta: Optional delta value
        help_text: Optional additional help text
    """
    st.metric(label=label, value=value, delta=delta, help=help_text)

    if tooltip_key in TOOLTIPS:
        with st.container():
            st.caption(TOOLTIPS[tooltip_key]['title'])


def render_info_box(title, content, icon='ℹ️', color='info'):
    """
    Render an informational box

    Args:
        title: Box title
        content: Box content (markdown)
        icon: Icon to display
        color: Box type (info, success, warning, error)
    """
    color_map = {
        'info': '🔵',
        'success': '🟢',
        'warning': '🟡',
        'error': '🔴'
    }

    emoji = color_map.get(color, 'ℹ️')

    st.markdown(f"""
    <div style='padding: 1rem; border-radius: 0.5rem; background-color: #2D2D2D;
                border-left: 4px solid {"#4ECDC4" if color == "info" else "#FFD700" if color == "warning" else "#FF6B6B"}; margin: 1rem 0;'>
        <strong>{emoji} {title}</strong><br/>
        {content}
    </div>
    """, unsafe_allow_html=True)


def render_glossary():
    """Render a glossary of all terms"""
    st.markdown("### 📖 Glossary of Terms")

    categories = {
        '📊 Performance Metrics': ['diamond_hand_score', 'three_x_plus_rate', 'win_rate', 'avg_hold_time_hours', 'total_pnl_sol', 'consistency_score'],
        '🤖 ML Predictions': ['token_3x_probability', 'model_auc'],
        '📈 Risk Metrics': ['sharpe_ratio', 'max_drawdown'],
        '🏷️ Trading Patterns': ['diamond_hand', 'scalper', 'sniper']
    }

    for category, keys in categories.items():
        with st.expander(category, expanded=False):
            for key in keys:
                if key in TOOLTIPS:
                    tooltip = TOOLTIPS[key]
                    st.markdown(f"**{tooltip['title']}**")
                    st.markdown(tooltip['content'])
                    st.markdown("---")
