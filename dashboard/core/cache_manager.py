"""
Cache Manager for Dashboard

Implements multi-layer caching strategy:
- L1: Streamlit cache (in-memory, TTL-based)
- L2: Disk cache (persistent across sessions)
- L3: Database query cache (materialized views)
"""

import streamlit as st
import hashlib
import json
import pickle
from functools import wraps
from typing import Any, Callable, Optional, Dict
from pathlib import Path
from datetime import timedelta

try:
    import diskcache as dc
    DISKCACHE_AVAILABLE = True
except ImportError:
    DISKCACHE_AVAILABLE = False
    print("Warning: diskcache not available, using memory cache only")


class CacheManager:
    """
    Multi-layer cache manager for dashboard performance optimization.

    Layers:
    1. Streamlit @st.cache_data - Fast, in-memory, session-bound
    2. Disk cache - Persistent, survives session restarts
    3. Query cache - Materialized views in database
    """

    def __init__(self, cache_dir: str = 'data/cache'):
        """
        Initialize cache manager

        Args:
            cache_dir: Directory for disk cache
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize disk cache if available
        if DISKCACHE_AVAILABLE:
            self.disk_cache = dc.Cache(str(self.cache_dir))
        else:
            self.disk_cache = None

    def _make_key(self, prefix: str, args: tuple, kwargs: dict) -> str:
        """
        Generate cache key from function arguments

        Args:
            prefix: Function name or custom prefix
            args: Function positional arguments
            kwargs: Function keyword arguments

        Returns:
            Cache key string
        """
        key_dict = {'args': args, 'kwargs': kwargs}
        key_str = json.dumps(key_dict, sort_keys=True, default=str)
        key_hash = hashlib.md5(key_str.encode()).hexdigest()
        return f"{prefix}:{key_hash}"

    def cached(self, ttl: int = 3600, use_disk: bool = True, use_streamlit: bool = True):
        """
        Decorator for multi-layer caching

        Args:
            ttl: Time to live in seconds
            use_disk: Whether to use disk cache
            use_streamlit: Whether to use Streamlit cache

        Returns:
            Decorated function with caching
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Try Streamlit cache first (fastest)
                if use_streamlit:
                    # Streamlit cache is handled by @st.cache_data decorator
                    # This is just a pass-through
                    pass

                # Try disk cache
                if use_disk and self.disk_cache is not None:
                    key = self._make_key(func.__name__, args, kwargs)
                    result = self.disk_cache.get(key)

                    if result is not None:
                        return result

                # Execute function
                result = func(*args, **kwargs)

                # Save to disk cache
                if use_disk and self.disk_cache is not None:
                    key = self._make_key(func.__name__, args, kwargs)
                    self.disk_cache.set(key, result, expire=ttl)

                return result

            return wrapper
        return decorator

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from disk cache

        Args:
            key: Cache key
            default: Default value if not found

        Returns:
            Cached value or default
        """
        if self.disk_cache is None:
            return default

        value = self.disk_cache.get(key)
        return value if value is not None else default

    def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """
        Set value in disk cache

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
        """
        if self.disk_cache is None:
            return

        self.disk_cache.set(key, value, expire=ttl)

    def delete(self, key: str) -> bool:
        """
        Delete value from disk cache

        Args:
            key: Cache key

        Returns:
            True if deleted, False if not found
        """
        if self.disk_cache is None:
            return False

        try:
            del self.disk_cache[key]
            return True
        except KeyError:
            return False

    def clear(self) -> None:
        """Clear all disk cache"""
        if self.disk_cache is None:
            return

        self.disk_cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics

        Returns:
            Dictionary with cache stats
        """
        if self.disk_cache is None:
            return {
                'disk_cache_available': False,
                'size': 0,
                'count': 0
            }

        return {
            'disk_cache_available': True,
            'size': self.disk_cache.volume(),  # Size in bytes
            'count': len(self.disk_cache)
        }


class StreamlitCacheHelper:
    """
    Helper for Streamlit's @st.cache_data decorator.

    Provides convenient methods with automatic TTL and versioning.
    """

    @staticmethod
    def cache_data(ttl: int = 300, max_entries: int = 100):
        """
        Streamlit cache decorator with sensible defaults

        Args:
            ttl: Time to live in seconds (default: 5 min)
            max_entries: Maximum number of entries in cache

        Returns:
            Decorator function
        """
        return st.cache_data(ttl=ttl, max_entries=max_entries)

    @staticmethod
    def cache_hash(max_entries: int = 100):
        """
        Streamlit cache with hash exclusion (for unhashable args)

        Args:
            max_entries: Maximum number of entries in cache

        Returns:
            Decorator function
        """
        return st.cache_data(max_entries=max_entries, show_spinner=False)


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """
    Get or create global CacheManager instance

    Returns:
        CacheManager instance
    """
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


# Convenience decorators

def cached(ttl: int = 3600, use_disk: bool = True):
    """
    Convenience decorator for caching functions

    Args:
        ttl: Time to live in seconds (default: 1 hour)
        use_disk: Whether to use persistent disk cache

    Example:
        @cached(ttl=600)
        def expensive_function(param):
            return costly_computation(param)
    """
    return get_cache_manager().cached(ttl=ttl, use_disk=use_disk)


def streamlit_cached(ttl: int = 300, max_entries: int = 100):
    """
    Convenience decorator for Streamlit caching

    Args:
        ttl: Time to live in seconds (default: 5 min)
        max_entries: Maximum cache entries

    Example:
        @streamlit_cached(ttl=60)
        def load_leaderboard():
            return get_leaderboard_data()
    """
    return StreamlitCacheHelper.cache_data(ttl=ttl, max_entries=max_entries)


def invalidate_all_cache():
    """Invalidate all disk cache"""
    manager = get_cache_manager()
    manager.clear()


def get_cache_stats() -> Dict[str, Any]:
    """
    Get cache statistics

    Returns:
        Dictionary with cache stats
    """
    manager = get_cache_manager()
    return manager.get_stats()
