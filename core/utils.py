"""
Utility functions for KOL Tracker ML
"""

import asyncio
import time
from typing import List, Callable, Any
from datetime import datetime, timedelta


def retry_async(
    func: Callable,
    max_retries: int = 3,
    delay: float = 1.0,
    exceptions: tuple = (Exception,)
):
    """
    Retry decorator for async functions

    Args:
        func: Async function to retry
        max_retries: Maximum number of retries
        delay: Delay between retries in seconds
        exceptions: Tuple of exceptions to catch
    """
    async def wrapper(*args, **kwargs):
        for attempt in range(max_retries):
            try:
                return await func(*args, **kwargs)
            except exceptions as e:
                if attempt == max_retries - 1:
                    raise
                print(f"[!] Retry {attempt + 1}/{max_retries}: {e}")
                await asyncio.sleep(delay)
    return wrapper


def format_sol(amount: float) -> str:
    """
    Format SOL amount with appropriate precision

    Args:
        amount: SOL amount

    Returns:
        Formatted string
    """
    if abs(amount) >= 1000:
        return f"{amount:,.2f} SOL"
    elif abs(amount) >= 1:
        return f"{amount:.4f} SOL"
    else:
        return f"{amount:.6f} SOL"


def format_time_ago(timestamp: datetime) -> str:
    """
    Format timestamp as "X time ago"

    Args:
        timestamp: Datetime object

    Returns:
        Formatted string like "2 hours ago"
    """
    delta = datetime.utcnow() - timestamp

    seconds = delta.total_seconds()

    if seconds < 60:
        return f"{int(seconds)} seconds ago"
    elif seconds < 3600:
        return f"{int(seconds / 60)} minutes ago"
    elif seconds < 86400:
        return f"{int(seconds / 3600)} hours ago"
    elif seconds < 604800:
        return f"{int(seconds / 86400)} days ago"
    elif seconds < 2592000:
        return f"{int(seconds / 604800)} weeks ago"
    else:
        return f"{int(seconds / 2592000)} months ago"


def format_hold_time(seconds: float) -> str:
    """
    Format hold time in human-readable format

    Args:
        seconds: Hold time in seconds

    Returns:
        Formatted string
    """
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.1f}h"
    else:
        days = seconds / 86400
        return f"{days:.1f}d"


def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Format as percentage

    Args:
        value: Value between 0 and 1
        decimals: Number of decimal places

    Returns:
        Formatted string
    """
    return f"{value * 100:.{decimals}f}%"


def truncate_address(address: str, chars: int = 8) -> str:
    """
    Truncate wallet address for display

    Args:
        address: Full address
        chars: Number of characters to show at each end

    Returns:
        Truncated address like "CyaE1Vxv...a54o"
    """
    if len(address) < chars * 2:
        return address
    return f"{address[:chars]}...{address[-chars:]}"


def progress_bar(current: int, total: int, width: int = 50, prefix: str = "") -> str:
    """
    Generate progress bar string

    Args:
        current: Current progress
        total: Total value
        width: Width of progress bar
        prefix: Prefix string

    Returns:
        Progress bar string
    """
    if total == 0:
        percent = 100
    else:
        percent = (current / total) * 100

    filled = int(width * current / total) if total > 0 else 0
    bar = "█" * filled + "░" * (width - filled)

    return f"{prefix}[{bar}] {percent:.1f}% ({current}/{total})"


class Timer:
    """
    Simple context manager for timing code execution
    """

    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        print(f"[*] {self.name} completed in {elapsed:.2f} seconds")

    @property
    def elapsed(self) -> float:
        """Get elapsed time if timer is running"""
        if self.start_time is None:
            return 0
        end = self.end_time or time.time()
        return end - self.start_time


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")

    print("\n--- Format SOL ---")
    print(f"1000 SOL: {format_sol(1000)}")
    print(f"10.5 SOL: {format_sol(10.5)}")
    print(f"0.001 SOL: {format_sol(0.001)}")

    print("\n--- Format Time Ago ---")
    print(f"30 seconds ago: {format_time_ago(datetime.utcnow() - timedelta(seconds=30))}")
    print(f"2 hours ago: {format_time_ago(datetime.utcnow() - timedelta(hours=2))}")
    print(f"3 days ago: {format_time_ago(datetime.utcnow() - timedelta(days=3))}")

    print("\n--- Format Hold Time ---")
    print(f"45 seconds: {format_hold_time(45)}")
    print(f"15 minutes: {format_hold_time(900)}")
    print(f"6 hours: {format_hold_time(21600)}")
    print(f"2 days: {format_hold_time(172800)}")

    print("\n--- Progress Bar ---")
    for i in range(0, 101, 10):
        print(f"\r{progress_bar(i, 100)}", end="")
    print()

    print("\n--- Timer ---")
    with Timer("Test operation"):
        sum(range(1000000))
