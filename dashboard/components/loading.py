"""
Loading States Component

Provides enhanced loading indicators, progress bars, and skeleton screens.
"""

import streamlit as st
import time


def loading_spinner(message="Loading...", size="medium"):
    """
    Display a loading spinner with custom message

    Args:
        message: Message to display
        size: Size of spinner (small, medium, large)
    """
    size_map = {
        'small': '0.75rem',
        'medium': '1rem',
        'large': '1.5rem'
    }

    font_size = size_map.get(size, '1rem')

    st.markdown(f"""
    <div style='display: flex; justify-content: center; align-items: center; padding: 3rem; background: var(--bg-tertiary); border-radius: 0.75rem; margin: 1rem 0;'>
        <div style='text-align: center;'>
            <div class='spinner' style='width: 40px; height: 40px; border-width: 4px; margin: 0 auto 1rem;'></div>
            <p style='margin: 0; color: var(--text-secondary); font-size: {font_size};'>{message}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)


def skeleton_card(height="100px", width="100%"):
    """
    Render a skeleton loading card

    Args:
        height: Height of skeleton
        width: Width of skeleton
    """
    st.markdown(f"""
    <div class='skeleton' style='height: {height}; width: {width};'></div>
    """, unsafe_allow_html=True)


def skeleton_row(count=4, height="80px"):
    """
    Render multiple skeleton cards in a row

    Args:
        count: Number of skeleton cards
        height: Height of each card
    """
    html = "<div style='display: flex; gap: 1rem; margin: 1rem 0;'>"
    for i in range(count):
        html += f"<div class='skeleton' style='height: {height}; flex: 1; border-radius: 0.5rem;'></div>"
    html += "</div>"

    st.markdown(html, unsafe_allow_html=True)


def progress_with_message(progress, message, status=None):
    """
    Display progress bar with status message

    Args:
        progress: Progress value (0-1)
        message: Status message
        status: Optional status (info, success, warning, error)
    """
    st.progress(progress)

    if status:
        status_map = {
            'info': ('🔵', '#667EEA'),
            'success': ('🟢', '#4ECDC4'),
            'warning': ('🟡', '#FFD700'),
            'error': ('🔴', '#FF6B6B')
        }
        emoji, color = status_map.get(status, ('ℹ️', '#4ECDC4'))

        st.markdown(f"""
        <div style='color: {color}; font-size: 0.9rem; margin-top: 0.5rem;'>
            {emoji} {message}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.caption(message)


def animated_counter(final_value, label="", duration=1.0, prefix="", suffix=""):
    """
    Display an animated counter (placeholder - Streamlit doesn't support JS animations)

    Args:
        final_value: Final value to display
        label: Optional label
        duration: Animation duration (not applicable in Streamlit)
        prefix: Prefix for value
        suffix: Suffix for value
    """
    st.markdown(f"""
    <div class='metric-card scale-in'>
        <div style='color: var(--text-secondary); font-size: 0.875rem; margin-bottom: 0.5rem;'>{label}</div>
        <div style='font-size: 2rem; font-weight: 700; color: var(--text-primary);'>
            {prefix}{final_value}{suffix}
        </div>
    </div>
    """, unsafe_allow_html=True)


def staged_progress(stages, current_stage):
    """
    Display staged progress indicator

    Args:
        stages: List of stage names
        current_stage: Index of current stage
    """
    html = "<div style='display: flex; align-items: center; justify-content: space-between; margin: 2rem 0;'>"

    for i, stage in enumerate(stages):
        is_completed = i < current_stage
        is_current = i == current_stage

        if is_completed:
            bg_color = "var(--success)"
            border_color = "var(--success)"
            text_color = "white"
        elif is_current:
            bg_color = "var(--accent-primary)"
            border_color = "var(--accent-primary)"
            text_color = "white"
        else:
            bg_color = "var(--bg-tertiary)"
            border_color = "var(--border)"
            text_color = "var(--text-secondary)"

        html += f"""
        <div style='display: flex; flex-direction: column; align-items: center; flex: 1;'>
            <div style='width: 40px; height: 40px; border-radius: 50%; background: {bg_color};
                       border: 3px solid {border_color}; display: flex; align-items: center;
                       justify-content: center; color: {text_color}; font-weight: 700;
                       margin-bottom: 0.5rem;'>
                {i + 1}
            </div>
            <div style='font-size: 0.75rem; color: {text_color}; text-align: center;'>
                {stage}
            </div>
        </div>
        """

        if i < len(stages) - 1:
            line_color = "var(--success)" if i < current_stage else "var(--border)"
            html += f"<div style='height: 2px; flex: 0.5; background: {line_color}; margin: 0 0.5rem 2.5rem;'></div>"

    html += "</div>"

    st.markdown(html, unsafe_allow_html=True)


def error_boundary(message="Something went wrong", details=None):
    """
    Display an error boundary with optional details

    Args:
        message: Error message
        details: Optional error details
    """
    st.markdown(f"""
    <div class='error-box'>
        <strong>⚠️ {message}</strong>
        {f"<br/><details><summary>Error Details</summary><pre>{details}</pre></details>" if details else ""}
    </div>
    """, unsafe_allow_html=True)


def empty_state(icon="📭", title="No Data", message="There's nothing to display yet"):
    """
    Display an empty state with helpful message

    Args:
        icon: Icon to display
        title: Title of empty state
        message: Helpful message
    """
    st.markdown(f"""
    <div style='text-align: center; padding: 3rem; background: var(--bg-tertiary); border-radius: 0.75rem; margin: 2rem 0;'>
        <div style='font-size: 4rem; margin-bottom: 1rem;'>{icon}</div>
        <h3 style='color: var(--text-primary); margin-bottom: 0.5rem;'>{title}</h3>
        <p style='color: var(--text-secondary);'>{message}</p>
    </div>
    """, unsafe_allow_html=True)


def success_alert(title, message, dismissible=False):
    """
    Display a success alert

    Args:
        title: Alert title
        message: Alert message
        dismissible: Whether alert can be dismissed (not supported in Streamlit)
    """
    st.markdown(f"""
    <div class='success-box'>
        <strong>✅ {title}</strong><br/>
        {message}
    </div>
    """, unsafe_allow_html=True)


def info_alert(title, message):
    """
    Display an info alert

    Args:
        title: Alert title
        message: Alert message
    """
    st.markdown(f"""
    <div class='info-box'>
        <strong>ℹ️ {title}</strong><br/>
        {message}
    </div>
    """, unsafe_allow_html=True)


def warning_alert(title, message):
    """
    Display a warning alert

    Args:
        title: Alert title
        message: Alert message
    """
    st.markdown(f"""
    <div class='warning-box'>
        <strong>⚠️ {title}</strong><br/>
        {message}
    </div>
    """, unsafe_allow_html=True)


@st.cache_data(ttl=60)
def load_data_with_fallback(data_loader, fallback_value=None):
    """
    Load data with fallback on error

    Args:
        data_loader: Function to load data
        fallback_value: Fallback value if loading fails

    Returns:
        Loaded data or fallback value
    """
    try:
        return data_loader()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return fallback_value
