"""
State Manager for Dashboard

Manages global dashboard state including filters, selections,
presets, and user preferences using Streamlit's session_state.
"""

import streamlit as st
from typing import Dict, List, Any, Optional
from datetime import datetime
import json


class DashboardState:
    """
    Manages dashboard state with persistence across page reruns.

    Features:
    - Filter management (get, set, update, clear)
    - KOL selection tracking
    - Preset management
    - User preferences (theme, auto-refresh, etc.)
    - Cache version control for invalidation
    """

    def __init__(self):
        """Initialize dashboard state if not exists"""
        if 'dashboard_state' not in st.session_state:
            st.session_state.dashboard_state = self._get_default_state()

    @staticmethod
    def _get_default_state() -> Dict[str, Any]:
        """Get default state structure"""
        return {
            # Filters
            'filters': {
                'min_diamond_score': 0,
                'min_trades': 1,
                'max_trades': 10000,
                'win_rate_range': [0.0, 1.0],
                'hold_time_range': [0.0, 1000.0],
                'dex_list': ['raydium', 'jupiter', 'orca', 'pump_fun'],
                'kol_type': 'all',  # all, diamond_hands, scalpers
                'date_range': None,  # [start_date, end_date]
                'token_filter': '',
                'search_query': ''
            },

            # Selections
            'selected_kols': [],  # List of KOL IDs
            'selected_tokens': [],  # List of token addresses
            'compared_kols': [],  # KOLs currently being compared

            # Presets
            'presets': [],  # List of saved preset dicts

            # User preferences
            'preferences': {
                'theme': 'tech_dark',  # dark, light, tech_dark, cyberpunk
                'default_page': 'leaderboard',
                'auto_refresh': False,
                'refresh_interval': 30,  # seconds
                'page_size': 25,
                'show_tooltips': True,
                'compact_mode': False
            },

            # Cache management
            'cache_version': 1,
            'last_data_update': None,

            # UI state
            'current_page': 'leaderboard',
            'sidebar_expanded': True,
            'filters_visible': True
        }

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from state

        Args:
            key: Dot-notation key (e.g., 'filters.min_diamond_score')
            default: Default value if key not found

        Returns:
            Value or default
        """
        # Check if dashboard_state exists
        if not hasattr(st.session_state, 'dashboard_state'):
            return default

        state = st.session_state.dashboard_state

        # Support dot notation
        keys = key.split('.')
        value = state
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def set(self, key: str, value: Any) -> None:
        """
        Set a value in state

        Args:
            key: Dot-notation key (e.g., 'filters.min_diamond_score')
            value: Value to set
        """
        state = st.session_state.dashboard_state

        # Support dot notation
        keys = key.split('.')
        current = state
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        current[keys[-1]] = value

        # Mark state as modified
        st.session_state.dashboard_state = state

    def update_filters(self, **filters) -> None:
        """
        Update multiple filters at once

        Args:
            **filters: Filter key-value pairs to update
        """
        current_filters = self.get('filters', {})
        current_filters.update(filters)
        self.set('filters', current_filters)

    def clear_filters(self) -> None:
        """Reset all filters to defaults"""
        default_state = self._get_default_state()
        self.set('filters', default_state['filters'])

    def add_selected_kol(self, kol_id: int) -> None:
        """Add a KOL to selection list"""
        selected = self.get('selected_kols', [])
        if kol_id not in selected:
            selected.append(kol_id)
            self.set('selected_kols', selected)

    def remove_selected_kol(self, kol_id: int) -> None:
        """Remove a KOL from selection list"""
        selected = self.get('selected_kols', [])
        if kol_id in selected:
            selected.remove(kol_id)
            self.set('selected_kols', selected)

    def toggle_selected_kol(self, kol_id: int) -> None:
        """Toggle KOL selection"""
        selected = self.get('selected_kols', [])
        if kol_id in selected:
            selected.remove(kol_id)
        else:
            selected.append(kol_id)
        self.set('selected_kols', selected)

    def clear_selected_kols(self) -> None:
        """Clear all KOL selections"""
        self.set('selected_kols', [])

    def save_preset(self, name: str, description: str = "") -> Dict[str, Any]:
        """
        Save current filters as a preset

        Args:
            name: Preset name
            description: Optional description

        Returns:
            Preset dictionary
        """
        preset = {
            'name': name,
            'description': description,
            'created_at': datetime.now().isoformat(),
            'page': self.get('current_page'),
            'filters': self.get('filters', {}),
            'sort': {
                'column': self.get('sort_column', 'diamond_hand_score'),
                'ascending': self.get('sort_ascending', False)
            }
        }

        presets = self.get('presets', [])
        presets.append(preset)
        self.set('presets', presets)

        return preset

    def load_preset(self, preset_name: str) -> bool:
        """
        Load a preset by name

        Args:
            preset_name: Name of preset to load

        Returns:
            True if loaded successfully, False otherwise
        """
        presets = self.get('presets', [])
        for preset in presets:
            if preset['name'] == preset_name:
                self.set('filters', preset['filters'])
                if 'sort' in preset:
                    self.set('sort_column', preset['sort']['column'])
                    self.set('sort_ascending', preset['sort']['ascending'])
                return True
        return False

    def delete_preset(self, preset_name: str) -> bool:
        """
        Delete a preset by name

        Args:
            preset_name: Name of preset to delete

        Returns:
            True if deleted, False if not found
        """
        presets = self.get('presets', [])
        original_length = len(presets)
        presets = [p for p in presets if p['name'] != preset_name]

        if len(presets) < original_length:
            self.set('presets', presets)
            return True
        return False

    def get_presets(self) -> List[Dict[str, Any]]:
        """Get all presets"""
        return self.get('presets', [])

    def set_preference(self, key: str, value: Any) -> None:
        """
        Set a user preference

        Args:
            key: Preference key
            value: Preference value
        """
        preferences = self.get('preferences', {})
        preferences[key] = value
        self.set('preferences', preferences)

    def get_preference(self, key: str, default: Any = None) -> Any:
        """
        Get a user preference

        Args:
            key: Preference key
            default: Default value if not found

        Returns:
            Preference value or default
        """
        preferences = self.get('preferences', {})
        return preferences.get(key, default)

    def increment_cache_version(self) -> int:
        """
        Increment cache version to invalidate all cached data

        Returns:
            New cache version
        """
        current_version = self.get('cache_version', 0)
        new_version = current_version + 1
        self.set('cache_version', new_version)
        return new_version

    def get_cache_version(self) -> int:
        """Get current cache version"""
        return self.get('cache_version', 0)

    def update_data_timestamp(self) -> None:
        """Update the last data update timestamp"""
        self.set('last_data_update', datetime.now().isoformat())

    def get_data_age_seconds(self) -> Optional[float]:
        """
        Get age of last data update in seconds

        Returns:
            Age in seconds, or None if no update timestamp
        """
        timestamp = self.get('last_data_update')
        if not timestamp:
            return None

        last_update = datetime.fromisoformat(timestamp)
        age = (datetime.now() - last_update).total_seconds()
        return age

    def export_state(self) -> str:
        """
        Export state as JSON string

        Returns:
            JSON string representation of state
        """
        return json.dumps(st.session_state.dashboard_state, indent=2, default=str)

    def import_state(self, state_json: str) -> bool:
        """
        Import state from JSON string

        Args:
            state_json: JSON string representation of state

        Returns:
            True if imported successfully, False otherwise
        """
        try:
            state = json.loads(state_json)
            st.session_state.dashboard_state = state
            return True
        except (json.JSONDecodeError, TypeError):
            return False

    def reset_to_defaults(self) -> None:
        """Reset all state to defaults"""
        st.session_state.dashboard_state = self._get_default_state()


# Global instance
_state_instance = None


def get_state() -> DashboardState:
    """
    Get or create global DashboardState instance

    Returns:
        DashboardState instance
    """
    global _state_instance
    if _state_instance is None:
        _state_instance = DashboardState()
    return _state_instance
