"""
Summary Scheduler - Generates automated summaries every 15 minutes

This script runs continuously and:
1. Generates comprehensive summaries every 15 minutes
2. Saves summaries to data/summaries/
3. Updates a latest summary file
4. Can trigger notifications

Usage:
    python run_summary_scheduler.py
"""

import asyncio
import time
from datetime import datetime
from pathlib import Path
import json

from dashboard.core.summary_generator import get_summary_generator


# Configuration
SUMMARY_INTERVAL_MINUTES = 15  # Generate summary every 15 minutes
SUMMARY_INTERVAL_SECONDS = SUMMARY_INTERVAL_MINUTES * 60


async def generate_and_save_summary():
    """
    Generate and save a summary

    Returns:
        Path to saved summary file
    """
    print("\n" + "="*60)
    print(f"GENERATING SUMMARY - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60 + "\n")

    try:
        # Get generator
        generator = get_summary_generator()

        # Generate summary
        print("Generating summary...")
        summary = generator.generate_summary()

        # Save summary
        print("Saving summary...")
        filepath = generator.save_summary(summary)
        print(f"Summary saved to: {filepath}")

        # Save as latest
        latest_path = Path('data/summaries/latest_summary.json')
        with open(latest_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Latest summary updated")

        # Generate markdown
        markdown = generator.get_summary_markdown(summary)
        markdown_path = Path('data/summaries/latest_summary.md')
        with open(markdown_path, 'w', encoding='utf-8') as f:
            f.write(markdown)
        print(f"Markdown summary saved")

        # Print summary to console (with UTF-8 encoding)
        try:
            print("\n" + markdown)
        except UnicodeEncodeError:
            # If console doesn't support UTF-8, skip markdown printing
            print("(Markdown contains special characters - see file for details)")

        print("\n" + "="*60)
        print("SUMMARY COMPLETE")
        print("="*60 + "\n")

        return filepath

    except Exception as e:
        print(f"\nError generating summary: {e}")
        import traceback
        traceback.print_exc()
        return None


async def summary_scheduler_loop():
    """
    Main scheduler loop - runs forever
    """
    print("\n" + "="*60)
    print("SUMMARY SCHEDULER STARTED")
    print("="*60)
    print(f"Will generate summaries every {SUMMARY_INTERVAL_MINUTES} minutes")
    print(f"Interval: {SUMMARY_INTERVAL_SECONDS} seconds")
    print("Press Ctrl+C to stop")
    print("="*60 + "\n")

    # Generate first summary immediately
    await generate_and_save_summary()

    # Track run count
    run_count = 1

    try:
        while True:
            # Wait for next interval
            print(f"Next summary in {SUMMARY_INTERVAL_MINUTES} minutes...")
            print(f"   (Press Ctrl+C to stop)\n")

            await asyncio.sleep(SUMMARY_INTERVAL_SECONDS)

            # Generate next summary
            run_count += 1
            print(f"\nRun #{run_count}")
            await generate_and_save_summary()

    except KeyboardInterrupt:
        print("\n\nSummary scheduler stopped by user")
    except Exception as e:
        print(f"\n\nError in scheduler: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Entry point"""
    # Create summaries directory
    summaries_dir = Path('data/summaries')
    summaries_dir.mkdir(parents=True, exist_ok=True)

    # Run scheduler
    asyncio.run(summary_scheduler_loop())


if __name__ == "__main__":
    main()
