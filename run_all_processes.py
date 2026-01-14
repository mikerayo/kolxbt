#!/usr/bin/env python3
"""
KOL Tracker ML - Orchestrator
Runs all processes in parallel from a single service

Processes:
- Tracker: Scans KOL trades every 5 min
- ML Trainer: Trains models every 6 hours
- Token Discovery: Discovers new traders every 12 hours
- Token Updater: Updates token metadata every 35 min
"""

import asyncio
import signal
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from processes.run_tracker_continuous import continuous_tracking_loop
from processes.run_continuous_trainer import main as trainer_main
from discovery.run_discovery_continuous import main as discovery_main
from processes.run_token_updater_both_continuous import main as token_updater_main


# Store running tasks
running_tasks = []


def signal_handler(sig, frame):
    """Handle shutdown signals"""
    print("\n" + "=" * 70)
    print("SHUTTING DOWN ALL PROCESSES...")
    print("=" * 70)

    for task in running_tasks:
        if not task.done():
            task.cancel()

    print("\n[+] All processes stopped")
    sys.exit(0)


async def run_tracker():
    """Run tracker continuously"""
    print("[*] Starting Tracker process...")
    await continuous_tracking_loop()


async def run_trainer():
    """Run ML trainer continuously"""
    print("[*] Starting ML Trainer process...")
    # Run in thread since it's a blocking function
    import asyncio
    await asyncio.to_thread(trainer_main)


async def run_discovery():
    """Run token discovery continuously"""
    print("[*] Starting Token Discovery process...")
    # Run in thread since it's a blocking function
    import asyncio
    await asyncio.to_thread(discovery_main)


async def run_token_updater():
    """Run token updater continuously"""
    print("[*] Starting Token Updater process...")
    # Token updater runs in its own loop
    import asyncio
    # Create a new task for the blocking main function
    # The main function runs forever in a loop
    await asyncio.to_thread(token_updater_main)


async def main():
    """Main orchestrator - runs all processes in parallel"""

    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Print banner
    print("\n" + "=" * 70)
    print("💎 KOL TRACKER ML - ORCHESTRATOR")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print("\n[*] Starting all processes in parallel...")
    print("-" * 70)

    # Create tasks for all processes
    global running_tasks
    running_tasks = [
        asyncio.create_task(run_tracker(), name="Tracker"),
        asyncio.create_task(run_trainer(), name="ML-Trainer"),
        asyncio.create_task(run_discovery(), name="Token-Discovery"),
        asyncio.create_task(run_token_updater(), name="Token-Updater"),
    ]

    print("\n[+] All processes started:")
    print("    🔍 Tracker          - Scans trades every 5 min")
    print("    🧠 ML Trainer       - Trains models every 6 hours")
    print("    🕵️ Token Discovery - Discovers traders every 12 hours")
    print("    🪙 Token Updater    - Updates metadata every 35 min")
    print("\n" + "=" * 70)
    print("ALL SYSTEMS OPERATIONAL")
    print("=" * 70)
    print("\n[*] Press Ctrl+C to stop all processes\n")

    # Wait for all tasks (they run forever)
    try:
        await asyncio.gather(*running_tasks)
    except asyncio.CancelledError:
        print("\n[!] Tasks cancelled, shutting down...")
    except Exception as e:
        print(f"\n[!] Error in main loop: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        signal_handler(None, None)
