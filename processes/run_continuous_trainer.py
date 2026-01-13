#!/usr/bin/env python3
"""
Continuous ML Training Runner
Runs automatic model retraining every N hours
"""

import sys
import argparse
import signal
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from continuous_trainer.auto_trainer import ContinuousTrainer


def signal_handler(sig, frame):
    """Handle interrupt signals"""
    print("\n[!] Interrupt received, stopping trainer...")
    trainer.stop()
    sys.exit(0)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Continuous ML Trainer for KOL Tracker')

    parser.add_argument(
        '--interval',
        type=int,
        default=6,
        help='Retrain interval in hours (default: 6)'
    )

    parser.add_argument(
        '--check-interval',
        type=int,
        default=300,
        help='Check interval in seconds (default: 300 = 5 minutes)'
    )

    parser.add_argument(
        '--once',
        action='store_true',
        help='Run training once and exit'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training (default: 32)'
    )

    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate (default: 0.001)'
    )

    parser.add_argument(
        '--min-data-age',
        type=int,
        default=24,
        help='Minimum data age in hours before retraining (default: 24)'
    )

    args = parser.parse_args()

    # Print banner
    print("=" * 70)
    print("KOL Tracker - Continuous ML Training")
    print("=" * 70)
    print(f"Start time: {datetime.now()}")
    print(f"Retrain interval: {args.interval} hours")
    print(f"Check interval: {args.check_interval} seconds")
    print(f"Mode: {'Run once' if args.once else 'Continuous'}")
    print("=" * 70)

    # Create trainer
    trainer = ContinuousTrainer(
        retrain_interval_hours=args.interval,
        min_data_age_hours=args.min_data_age
    )

    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run training
    training_kwargs = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate
    }

    if args.once:
        # Run once and exit
        print("\n[*] Running training once...")
        results = trainer.train_all_models(**training_kwargs)

        # Print results
        print("\n" + "=" * 70)
        print("Training Results")
        print("=" * 70)

        for model_name, stats in results.items():
            if stats and stats.get('best_metric'):
                print(f"\n{model_name}:")
                print(f"  Best metric: {stats['best_metric']:.4f}")
                print(f"  Best epoch: {stats['best_epoch']}")
            else:
                print(f"\n{model_name}: Failed to train")

        print("\n[+] Training complete - exiting")
        sys.exit(0)

    else:
        # Run continuous
        print("\n[*] Starting continuous training loop...")
        print("[*] Press Ctrl+C to stop\n")

        # First training
        if trainer.should_retrain():
            trainer.train_all_models(**training_kwargs)

        # Run continuous loop
        trainer.run_continuous(check_interval_seconds=args.check_interval)


if __name__ == "__main__":
    main()
