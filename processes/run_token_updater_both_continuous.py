"""
Token Updater Continuous - DexScreener + Bubblemaps

Updates token metadata every 35 minutes with:
- DexScreener: Price, volume, liquidity, market cap
- Bubblemaps: Holder distribution, concentration metrics

Usage:
    python run_token_updater_both_continuous.py
"""

import asyncio
import signal
import sys
import io
from datetime import datetime
from pathlib import Path

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from updaters.update_tokens_both import update_tokens_with_bubblemaps
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('token_updater_both.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TokenUpdater:
    """Manages continuous token updates"""

    def __init__(self, interval_minutes=35):
        self.interval_minutes = interval_minutes
        self.interval_seconds = interval_minutes * 60
        self.running = False

    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info("Received shutdown signal...")
        self.running = False

    async def run(self):
        """Run continuous token updates"""
        self.running = True

        # Setup signal handlers (only works in main thread)
        try:
            signal.signal(signal.SIGINT, self.signal_handler)
            signal.signal(signal.SIGTERM, self.signal_handler)
        except ValueError:
            # Running in a thread (orchestrator mode), signals handled by parent
            pass

        logger.info("=" * 70)
        logger.info("TOKEN UPDATER - DexScreener + Bubblemaps")
        logger.info("=" * 70)
        logger.info(f"Start time: {datetime.now()}")
        logger.info(f"Update interval: {self.interval_minutes} minutes")
        logger.info(f"Tokens per update: 50")
        logger.info("=" * 70)
        logger.info("Starting continuous token update...")
        logger.info("Press Ctrl+C to stop\n")

        while self.running:
            try:
                logger.info("=" * 70)
                logger.info(f"Starting Token Update - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info("=" * 70)

                # Update tokens with both APIs
                result = await update_tokens_with_bubblemaps(limit=50)

                logger.info(f"Update completed:")
                logger.info(f"  DexScreener: {result['dex_found']} tokens")
                logger.info(f"  Bubblemaps: {result['bubble_found']} tokens")
                logger.info(f"  Created: {result['created']}")
                logger.info(f"  Updated: {result['updated']}")
                logger.info(f"  Errors: {result['errors']}")

                if self.running:
                    # Calculate next run time
                    from datetime import timedelta
                    next_run = datetime.now() + timedelta(minutes=self.interval_minutes)
                    logger.info(f"Next update in {self.interval_minutes} minutes "
                               f"(at {next_run.strftime('%Y-%m-%d %H:%M')})")

                    # Wait for interval
                    await asyncio.sleep(self.interval_seconds)

            except Exception as e:
                logger.error(f"Error in token updater: {e}")
                import traceback
                logger.error(traceback.format_exc())

                if self.running:
                    logger.info("Retrying in 5 minutes...")
                    await asyncio.sleep(300)

        logger.info("Token updater stopped")


def main():
    """Entry point"""
    updater = TokenUpdater(interval_minutes=35)
    asyncio.run(updater.run())


if __name__ == "__main__":
    main()
