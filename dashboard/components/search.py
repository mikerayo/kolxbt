"""
Fuzzy Search Component for Dashboard

Provides fast fuzzy search for KOLs and tokens using RapidFuzz.
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple

try:
    from rapidfuzz import process, fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    # Fallback to fuzzywuzzy
    try:
        from fuzzywuzzy import process, fuzz
        FUZZYWUZZY_FALLBACK = True
    except ImportError:
        FUZZYWUZZY_FALLBACK = False


class FuzzySearch:
    """
    Fast fuzzy search implementation using RapidFuzz.

    Features:
    - Search KOLs by name or wallet address
    - Search tokens by address
    - Configurable score threshold
    - Maximum results limit
    """

    def __init__(self, score_cutoff: int = 70):
        """
        Initialize fuzzy search

        Args:
            score_cutoff: Minimum similarity score (0-100)
        """
        self.score_cutoff = score_cutoff

        if not RAPIDFUZZ_AVAILABLE and not FUZZYWUZZY_FALLBACK:
            st.warning("⚠️ Fuzzy search not available. Install rapidfuzz: pip install rapidfuzz")

    def search_kols(
        self,
        query: str,
        kols_df,
        limit: int = 10,
        search_fields: List[str] = None
    ) -> pd.DataFrame:
        """
        Search KOLs by name or wallet address

        Args:
            query: Search query string
            kols_df: DataFrame with KOL data
            limit: Maximum number of results
            search_fields: Fields to search in (default: ['name', 'wallet_address'])

        Returns:
            DataFrame with matching KOLs
        """
        import pandas as pd

        if not RAPIDFUZZ_AVAILABLE and not FUZZYWUZZY_FALLBACK:
            # Fallback to simple contains search
            return self._simple_search(query, kols_df, limit, search_fields)

        if search_fields is None:
            search_fields = ['name', 'wallet_address']

        if kols_df.empty:
            return pd.DataFrame()

        results = []

        # Combine search fields for matching
        for _, kol in kols_df.iterrows():
            search_text = ' '.join([
                str(kol.get(field, ''))
                for field in search_fields
                if field in kol
            ])

            # Calculate similarity score
            score = fuzz.WRatio(query, search_text)

            if score >= self.score_cutoff:
                results.append({
                    **kol.to_dict(),
                    'match_score': score
                })

        # Sort by score and limit
        results.sort(key=lambda x: x['match_score'], reverse=True)
        results = results[:limit]

        return pd.DataFrame(results)

    def search_tokens(
        self,
        query: str,
        tokens_df,
        limit: int = 10,
        search_fields: List[str] = None
    ) -> pd.DataFrame:
        """
        Search tokens by address

        Args:
            query: Search query string
            tokens_df: DataFrame with token data
            limit: Maximum number of results
            search_fields: Fields to search in

        Returns:
            DataFrame with matching tokens
        """
        import pandas as pd

        if search_fields is None:
            search_fields = ['token_address', 'symbol']

        if tokens_df.empty:
            return pd.DataFrame()

        if not RAPIDFUZZ_AVAILABLE and not FUZZYWUZZY_FALLBACK:
            return self._simple_search(query, tokens_df, limit, search_fields)

        results = []

        for _, token in tokens_df.iterrows():
            search_text = ' '.join([
                str(token.get(field, ''))
                for field in search_fields
                if field in token
            ])

            score = fuzz.WRatio(query, search_text)

            if score >= self.score_cutoff:
                results.append({
                    **token.to_dict(),
                    'match_score': score
                })

        results.sort(key=lambda x: x['match_score'], reverse=True)
        results = results[:limit]

        return pd.DataFrame(results)

    def _simple_search(
        self,
        query: str,
        df,
        limit: int,
        search_fields: List[str]
    ) -> pd.DataFrame:
        """
        Simple contains search fallback

        Args:
            query: Search query
            df: DataFrame to search
            limit: Max results
            search_fields: Fields to search

        Returns:
            Filtered DataFrame
        """
        if df.empty:
            return df

        if search_fields is None:
            search_fields = df.columns.tolist()

        query_lower = query.lower()

        # Create mask for matching rows
        mask = pd.Series([False] * len(df), index=df.index)

        for field in search_fields:
            if field in df.columns:
                field_mask = df[field].astype(str).str.lower().str.contains(
                    query_lower, na=False
                )
                mask = mask | field_mask

        # Filter and limit
        filtered = df[mask]
        return filtered.head(limit)


def render_search_bar(
    placeholder: str = "🔍 Search KOLs by name or wallet...",
    key: str = "search"
) -> str:
    """
    Render a search bar component

    Args:
        placeholder: Text placeholder for search input
        key: Unique key for the component

    Returns:
        Search query string
    """
    query = st.text_input(
        placeholder,
        key=key,
        help="Type to search - fuzzy matching is enabled"
    )
    return query


def render_search_with_filters(
    data_source: str = "kols",
    filters: Dict[str, Any] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Render search bar with additional filters

    Args:
        data_source: Type of data ('kols' or 'tokens')
        filters: Dictionary of current filter values

    Returns:
        Tuple of (search_query, updated_filters)
    """
    search_query = render_search_bar(
        placeholder=f"🔍 Search {data_source}...",
        key=f"search_{data_source}"
    )

    updated_filters = filters.copy() if filters else {}

    # Add additional filters based on data source
    if data_source == "kols":
        col1, col2, col3 = st.columns(3)

        with col1:
            min_score = st.slider(
                "Min Diamond Hand Score",
                0, 100,
                int(updated_filters.get('min_diamond_score', 0)),
                key=f"filter_score_{data_source}"
            )
            updated_filters['min_diamond_score'] = min_score

        with col2:
            min_trades = st.number_input(
                "Min Trades",
                1, 1000,
                int(updated_filters.get('min_trades', 1)),
                key=f"filter_trades_{data_source}"
            )
            updated_filters['min_trades'] = min_trades

        with col3:
            kol_type = st.selectbox(
                "KOL Type",
                ['all', 'diamond_hands_only', 'scalpers_only'],
                index=['all', 'diamond_hands_only', 'scalpers_only'].index(
                    updated_filters.get('kol_type', 'all')
                ),
                key=f"filter_type_{data_source}"
            )
            updated_filters['kol_type'] = kol_type

    return search_query, updated_filters


def render_search_results(
    results_df,
    display_columns: List[str] = None,
    key_column: str = 'id'
):
    """
    Render search results in a nice format

    Args:
        results_df: DataFrame with search results
        display_columns: Columns to display
        key_column: Column to use as unique key
    """
    import pandas as pd

    if results_df.empty:
        st.info("🔍 No results found. Try a different search query.")
        return

    if display_columns is None:
        # Default columns to display
        if 'name' in results_df.columns:
            display_columns = ['name', 'diamond_hand_score', 'total_trades', 'win_rate']
        elif 'token_address' in results_df.columns:
            display_columns = ['token_address', 'trade_count', 'unique_kols', 'avg_multiple']
        else:
            display_columns = results_df.columns.tolist()[:5]

    # Format display
    display_df = results_df[display_columns].copy()

    # Format percentage columns
    if 'win_rate' in display_df.columns:
        display_df['win_rate'] = (display_df['win_rate'] * 100).round(1).astype(str) + '%'

    if 'match_score' in display_df.columns:
        display_df['match_score'] = display_df['match_score'].round(0).astype(int)

    # Display
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    # Show match count
    st.caption(f"Found {len(results_df)} results")


# Streamlit component wrapper
def fuzzy_search_component(
    data_source: str = "kols",
    data_df=None,
    on_select: callable = None
) -> Any:
    """
    Complete fuzzy search component with results display

    Args:
        data_source: Type of data ('kols' or 'tokens')
        data_df: DataFrame with data to search
        on_select: Callback function when item is selected

    Returns:
        Selected item or None
    """
    if data_df is None or data_df.empty:
        st.warning(f"No {data_source} data available")
        return None

    # Render search and filters
    query, filters = render_search_with_filters(data_source)

    # Perform search
    searcher = FuzzySearch(score_cutoff=70)

    if query:
        if data_source == "kols":
            results = searcher.search_kols(query, data_df, limit=20)
        else:
            results = searcher.search_tokens(query, data_df, limit=20)
    else:
        # No search query, return filtered data
        results = data_df

    # Apply additional filters
    if data_source == "kols" and filters:
        from dashboard.core.data_manager import get_data_manager
        data_manager = get_data_manager()
        results = data_manager.get_filtered_kols(
            min_score=filters.get('min_diamond_score', 0),
            min_trades=filters.get('min_trades', 1),
            kol_type=filters.get('kol_type', 'all')
        )

    # Display results
    if not results.empty:
        render_search_results(results)

        # Allow selection
        if on_select and 'name' in results.columns:
            selected_name = st.selectbox(
                "Select to view details",
                options=results['name'].tolist(),
                key=f"select_{data_source}"
            )

            if selected_name:
                selected_row = results[results['name'] == selected_name].iloc[0]
                if st.button("View Details", key=f"view_{data_source}"):
                    return on_select(selected_row)

    return None
