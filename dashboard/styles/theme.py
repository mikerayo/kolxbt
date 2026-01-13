"""
Theme Manager - Enhanced UI/UX with animations and responsive design
"""

import streamlit as st
from pathlib import Path


# Theme definitions
THEMES = {
    'dark': {
        'bg_primary': '#0E0E0E',
        'bg_secondary': '#1E1E1E',
        'bg_tertiary': '#2D2D2D',
        'text_primary': '#FFFFFF',
        'text_secondary': '#B0B0B0',
        'accent_primary': '#4ECDC4',
        'accent_secondary': '#FF6B6B',
        'accent_tertiary': '#FFD700',
        'border': '#444444',
        'success': '#4ECDC4',
        'warning': '#FFD700',
        'error': '#FF6B6B',
        'info': '#667EEA'
    },
    'tech_dark': {
        'bg_primary': '#1A1A1A',
        'bg_secondary': '#1E1E1E',
        'bg_tertiary': '#252525',
        'bg_hover': '#2A2A2A',
        'text_primary': '#FFFFFF',
        'text_secondary': '#B0B0B0',
        'text_tertiary': '#808080',
        'accent_primary': '#3B82F6',
        'accent_secondary': '#10B981',
        'accent_tertiary': '#F59E0B',
        'accent_error': '#EF4444',
        'border': '#2A2A2A',
        'success': '#10B981',
        'warning': '#F59E0B',
        'error': '#EF4444',
        'info': '#3B82F6'
    },
    'light': {
        'bg_primary': '#FFFFFF',
        'bg_secondary': '#F5F5F5',
        'bg_tertiary': '#E8E8E8',
        'text_primary': '#1A1A1A',
        'text_secondary': '#666666',
        'accent_primary': '#44A08D',
        'accent_secondary': '#FF8E53',
        'accent_tertiary': '#FFA500',
        'border': '#DDDDDD',
        'success': '#44A08D',
        'warning': '#FFA500',
        'error': '#FF6B6B',
        'info': '#667EEA'
    },
    'cyberpunk': {
        'bg_primary': '#0D0221',
        'bg_secondary': '#1A0A2E',
        'bg_tertiary': '#2D1B4E',
        'text_primary': '#00F0FF',
        'text_secondary': '#B026FF',
        'accent_primary': '#00F0FF',
        'accent_secondary': '#FF006E',
        'accent_tertiary': '#FFBE0B',
        'border': '#B026FF',
        'success': '#00F0FF',
        'warning': '#FFBE0B',
        'error': '#FF006E',
        'info': '#8338EC'
    }
}


def get_css(theme='dark', animations=True, responsive=True, enhanced=True):
    """
    Get complete CSS for the dashboard

    Args:
        theme: Theme name ('dark', 'light', 'tech_dark', 'cyberpunk')
        animations: Enable/disable animations
        responsive: Enable responsive design
        enhanced: Use enhanced professional theme

    Returns:
        CSS string
    """
    # Load tech_dark CSS if theme is tech_dark
    if theme == 'tech_dark':
        try:
            tech_dark_css_path = Path(__file__).parent / 'tech_dark.css'
            if tech_dark_css_path.exists():
                with open(tech_dark_css_path, 'r', encoding='utf-8') as f:
                    tech_dark_css = f.read()
                return tech_dark_css
        except Exception:
            pass

    # Load enhanced CSS if available
    if enhanced:
        try:
            enhanced_css_path = Path(__file__).parent / 'enhanced.css'
            if enhanced_css_path.exists():
                with open(enhanced_css_path, 'r', encoding='utf-8') as f:
                    enhanced_css = f.read()
                return enhanced_css
        except Exception:
            pass

    colors = THEMES.get(theme, THEMES['dark'])

    css = f"""
    <style>
    /* Base Theme Variables */
    :root {{
        --bg-primary: {colors['bg_primary']};
        --bg-secondary: {colors['bg_secondary']};
        --bg-tertiary: {colors['bg_tertiary']};
        --text-primary: {colors['text_primary']};
        --text-secondary: {colors['text_secondary']};
        --accent-primary: {colors['accent_primary']};
        --accent-secondary: {colors['accent_secondary']};
        --accent-tertiary: {colors['accent_tertiary']};
        --border: {colors['border']};
        --success: {colors['success']};
        --warning: {colors['warning']};
        --error: {colors['error']};
        --info: {colors['info']};
    }}

    /* Global Styles */
    .stApp {{
        background-color: var(--bg-primary);
        color: var(--text-primary);
    }}

    /* Main Container */
    .main .block-container {{
        background-color: var(--bg-secondary);
        border-radius: 1rem;
        padding: 2rem;
        max-width: 1400px;
        margin: 0 auto;
    }}

    /* Headers */
    h1, h2, h3, h4, h5, h6 {{
        color: var(--text-primary) !important;
        font-weight: 700;
    }}

    /* Metric Cards */
    .metric-card {{
        background: var(--bg-tertiary);
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 1px solid var(--border);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }}

    .metric-card:hover {{
        border-color: var(--accent-primary);
        box-shadow: 0 4px 12px rgba(78, 205, 196, 0.2);
        transform: translateY(-2px);
    }}

    /* Gradient Text */
    .gradient-text {{
        background: linear-gradient(90deg, var(--accent-secondary), var(--accent-primary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: var(--bg-secondary);
        border-right: 1px solid var(--border);
    }}

    /* Buttons */
    .stButton > button {{
        background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 0.5rem;
        color: white;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }}

    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(78, 205, 196, 0.4);
    }}

    /* Input Fields */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select,
    .stNumberInput > div > div > input {{
        background-color: var(--bg-tertiary);
        color: var(--text-primary);
        border: 1px solid var(--border);
        border-radius: 0.5rem;
        padding: 0.75rem;
    }}

    /* Dataframe */
    .stDataFrame {{
        border: 1px solid var(--border);
        border-radius: 0.5rem;
        overflow: hidden;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}

    .stTabs [data-baseweb="tab"] {{
        background-color: var(--bg-tertiary);
        border-radius: 0.5rem 0.5rem 0 0;
        padding: 1rem 1.5rem;
        transition: all 0.3s ease;
    }}

    .stTabs [aria-selected="true"] {{
        background-color: var(--accent-primary);
        color: white;
    }}

    /* Expander */
    .streamlit-expanderHeader {{
        background-color: var(--bg-tertiary);
        border: 1px solid var(--border);
        border-radius: 0.5rem;
        padding: 0.75rem;
        transition: all 0.3s ease;
    }}

    .streamlit-expanderHeader:hover {{
        border-color: var(--accent-primary);
    }}

    /* Progress Bar */
    .stProgress > div > div > div > div {{
        background: linear-gradient(90deg, var(--accent-primary), var(--accent-secondary));
    }}

    /* Info Boxes */
    .info-box {{
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: var(--bg-tertiary);
        border-left: 4px solid var(--accent-primary);
        margin: 1rem 0;
    }}

    .success-box {{
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: var(--bg-tertiary);
        border-left: 4px solid var(--success);
        margin: 1rem 0;
    }}

    .warning-box {{
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: var(--bg-tertiary);
        border-left: 4px solid var(--warning);
        margin: 1rem 0;
    }}

    .error-box {{
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: var(--bg-tertiary);
        border-left: 4px solid var(--error);
        margin: 1rem 0;
    }}

    /* Badges */
    .badge {{
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.875rem;
        font-weight: 600;
        margin: 0.25rem;
    }}

    .badge-diamond {{
        background: linear-gradient(135deg, var(--accent-primary), #44A08D);
        color: white;
    }}

    .badge-scalper {{
        background: linear-gradient(135deg, var(--accent-secondary), #FF8E53);
        color: white;
    }}

    .badge-sniper {{
        background: linear-gradient(135deg, #A8E6CF, #3DD5A6);
        color: white;
    }}

    /* Scrollbar */
    ::-webkit-scrollbar {{
        width: 10px;
        height: 10px;
    }}

    ::-webkit-scrollbar-track {{
        background: var(--bg-secondary);
    }}

    ::-webkit-scrollbar-thumb {{
        background: var(--border);
        border-radius: 5px;
    }}

    ::-webkit-scrollbar-thumb:hover {{
        background: var(--accent-primary);
    }}
    """

    # Add animations if enabled
    if animations:
        # Read animations.css file
        anim_path = Path(__file__).parent / 'animations.css'
        if anim_path.exists():
            with open(anim_path, 'r') as f:
                css += f.read()

    # Add responsive styles if enabled
    if responsive:
        css += """
    /* Responsive Design */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1rem;
        }

        .metric-card {
            padding: 1rem;
            margin-bottom: 0.75rem;
        }

        .gradient-text {
            font-size: 1.75rem;
        }

        h1 { font-size: 1.75rem; }
        h2 { font-size: 1.5rem; }
        h3 { font-size: 1.25rem; }

        .stDataFrame {
            font-size: 0.875rem;
        }

        .badge {
            font-size: 0.75rem;
            padding: 0.2rem 0.5rem;
        }

        /* Stack columns on mobile */
        .css-1d391kg {
            flex-direction: column !important;
        }
    }

    @media (max-width: 480px) {
        .main .block-container {
            padding: 0.75rem;
        }

        .metric-card {
            padding: 0.75rem;
        }

        .gradient-text {
            font-size: 1.5rem;
        }

        h1 { font-size: 1.5rem; }
        h2 { font-size: 1.25rem; }

        /* Hide less important elements on very small screens */
        .stSidebar .stMarkdown {
            font-size: 0.875rem;
        }
    }

    /* Loading State */
    .loading-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 3rem;
        background: var(--bg-tertiary);
        border-radius: 0.75rem;
        margin: 1rem 0;
    }

    .spinner {
        border: 4px solid var(--bg-tertiary);
        border-top: 4px solid var(--accent-primary);
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    """

    css += """
    </style>
    """

    return css


def render_theme(theme='dark', enhanced=True):
    """
    Render theme CSS to Streamlit

    Args:
        theme: Theme name ('dark', 'light', 'cyberpunk')
        enhanced: Use enhanced professional theme
    """
    css = get_css(theme=theme, animations=True, responsive=True, enhanced=enhanced)
    st.markdown(css, unsafe_allow_html=True)


def apply_theme_from_state():
    """Apply theme based on session state"""
    try:
        from dashboard.core.state_manager import get_state

        state = get_state()
        theme = state.get_preference('theme', 'dark')
        render_theme(theme=theme)
    except Exception:
        # If state is not initialized yet, use default theme
        render_theme(theme='dark')


def get_color(color_name, theme='dark'):
    """
    Get a specific color from the theme

    Args:
        color_name: Color key (e.g., 'accent_primary', 'bg_secondary')
        theme: Theme name

    Returns:
        Hex color code
    """
    colors = THEMES.get(theme, THEMES['dark'])
    return colors.get(color_name, '#FFFFFF')


def create_badge(text, badge_type='default', icon=''):
    """
    Create a badge HTML element

    Args:
        text: Badge text
        badge_type: Type (diamond, scalper, sniper, default)
        icon: Optional icon

    Returns:
        HTML string
    """
    badge_class = f'badge-{badge_type}' if badge_type != 'default' else 'badge'
    icon_html = f'{icon} ' if icon else ''

    return f'<span class="badge {badge_class}">{icon_html}{text}</span>'


def create_metric_card(label, value, delta=None, tooltip=None, icon=''):
    """
    Create an enhanced metric card

    Args:
        label: Card label
        value: Card value
        delta: Optional delta value
        tooltip: Optional tooltip text
        icon: Optional icon

    Returns:
        HTML string
    """
    icon_html = f'<div style="font-size: 2rem;">{icon}</div>' if icon else ''

    delta_html = ''
    if delta is not None:
        delta_color = '#4ECDC4' if delta > 0 else '#FF6B6B'
        delta_prefix = '+' if delta > 0 else ''
        delta_html = f'<div style="color: {delta_color}; font-size: 0.875rem;">{delta_prefix}{delta}</div>'

    tooltip_attr = f'title="{tooltip}"' if tooltip else ''

    return f'''
    <div class="metric-card" {tooltip_attr}>
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="color: var(--text-secondary); font-size: 0.875rem; margin-bottom: 0.5rem;">{label}</div>
                <div style="font-size: 2rem; font-weight: 700; color: var(--text-primary);">{value}</div>
                {delta_html}
            </div>
            {icon_html}
        </div>
    </div>
    '''


def render_loading_spinner(message="Loading..."):
    """
    Render a loading spinner

    Args:
        message: Optional message to display
    """
    st.markdown(f"""
    <div class="loading-container">
        <div style="text-align: center;">
            <div class="spinner"></div>
            <p style="margin-top: 1rem; color: var(--text-secondary);">{message}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
