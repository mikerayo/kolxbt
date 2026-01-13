"""
Continuous Auto-Trainer for ML Models
Automatically retrains models every N hours
"""

import time
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Optional
import threading
import torch

from deep_learning.token_predictor import TokenPredictor, create_token_predictor
from deep_learning.kol_predictor import KOLPredictor, create_kol_predictor
from deep_learning.training_pipeline import TokenPredictorTrainer, KOLPredictorTrainer
from deep_learning.data_loader import TokenPredictionDataset, KOLPerformanceDataset
from continuous_trainer.model_registry import ModelRegistry


class ContinuousTrainer:
    """
    Continuously retrains ML models every N hours
    """

    def __init__(
        self,
        retrain_interval_hours: int = 6,
        min_data_age_hours: int = 24,
        registry_path: str = 'models/model_registry.json',
        models_dir: str = 'models'
    ):
        """
        Initialize continuous trainer

        Args:
            retrain_interval_hours: Hours between retraining
            min_data_age_hours: Minimum data age before retraining
            registry_path: Path to model registry
            models_dir: Directory to save models
        """
        self.retrain_interval = retrain_interval_hours * 3600  # Convert to seconds
        self.min_data_age = min_data_age_hours
        self.registry_path = registry_path
        self.models_dir = models_dir
        self.last_train_time = None
        self.running = False

        # Load registry
        self.registry = ModelRegistry(registry_path)

        # Load last train time
        self._load_state()

    def _load_state(self):
        """Load training state from file"""
        state_path = 'models/trainer_state.json'
        if os.path.exists(state_path):
            try:
                with open(state_path, 'r') as f:
                    state = json.load(f)
                    self.last_train_time = state.get('last_train_time')
                    if self.last_train_time:
                        self.last_train_time = datetime.fromisoformat(self.last_train_time)
            except Exception as e:
                print(f"[!] Error loading state: {e}")

    def _save_state(self):
        """Save training state to file"""
        state_path = 'models/trainer_state.json'
        os.makedirs(os.path.dirname(state_path), exist_ok=True)

        state = {
            'last_train_time': self.last_train_time.isoformat() if self.last_train_time else None,
            'saved_at': datetime.now().isoformat()
        }

        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)

    def should_retrain(self) -> bool:
        """
        Check if it's time to retrain

        Returns:
            True if should retrain
        """
        if self.last_train_time is None:
            return True

        time_since_last = (datetime.now() - self.last_train_time).total_seconds()
        return time_since_last >= self.retrain_interval

    def train_token_model(
        self,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001
    ) -> Dict:
        """
        Train token predictor model

        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate

        Returns:
            Training statistics
        """
        print("\n" + "=" * 70)
        print("Training Token Predictor Model")
        print("=" * 70)

        # Determine date ranges
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)  # Use 60 days of data

        # Split into train/val (80/20)
        total_days = (end_date - start_date).days
        val_start_date = start_date + timedelta(days=int(total_days * 0.8))

        print(f"[*] Data range: {start_date.date()} to {end_date.date()}")
        print(f"    Train: {start_date.date()} to {val_start_date.date()}")
        print(f"    Val: {val_start_date.date()} to {end_date.date()}")

        # Create datasets
        try:
            train_dataset = TokenPredictionDataset(start_date, val_start_date, min_trades_per_kol=3)
            val_dataset = TokenPredictionDataset(val_start_date, end_date, min_trades_per_kol=3)

            if len(train_dataset) == 0 or len(val_dataset) == 0:
                print("[!] Not enough data for training")
                return {}

            # Create data loaders
            from torch.utils.data import DataLoader

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False
            )

            print(f"[+] Train samples: {len(train_dataset)}")
            print(f"[+] Val samples: {len(val_dataset)}")

            # Create model
            input_dim = 7  # From data_loader
            model = create_token_predictor(input_dim, model_type='mlp')

            # Create trainer
            trainer = TokenPredictorTrainer(
                model,
                learning_rate=learning_rate
            )

            # Train
            save_path = f'{self.models_dir}/token_predictor_best.pth'
            stats = trainer.train(
                train_loader,
                val_loader,
                epochs=epochs,
                save_path=save_path
            )

            # Register model
            version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.registry.register_model(
                'token_predictor',
                version,
                metrics=stats['val_metrics'][-1],
                metadata={
                    'train_samples': len(train_dataset),
                    'val_samples': len(val_dataset),
                    'epochs_trained': stats['best_epoch'],
                    'best_metric': stats['best_metric']
                },
                model_path=save_path
            )

            print(f"\n[+] Token predictor training complete")
            print(f"    Best AUC: {stats['best_metric']:.4f}")

            return stats

        except Exception as e:
            print(f"[!] Error training token predictor: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def train_kol_model(
        self,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001
    ) -> Dict:
        """
        Train KOL predictor model

        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate

        Returns:
            Training statistics
        """
        print("\n" + "=" * 70)
        print("Training KOL Predictor Model")
        print("=" * 70)

        # Prediction date (use past date to have future data)
        prediction_date = datetime.now() - timedelta(days=35)

        print(f"[*] Prediction date: {prediction_date.date()}")
        print(f"    History: 90 days")
        print(f"    Future: 7d and 30d")

        # Create datasets
        try:
            # We'll split the KOLs for train/val
            dataset = KOLPerformanceDataset(
                prediction_date,
                history_days=90,
                min_trades_per_kol=5
            )

            if len(dataset) == 0:
                print("[!] Not enough data for training")
                return {}

            # Simple train/val split (80/20)
            from torch.utils.data import random_split, DataLoader

            total_size = len(dataset)
            train_size = int(0.8 * total_size)
            val_size = total_size - train_size

            train_dataset, val_dataset = random_split(
                dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False
            )

            print(f"[+] Train KOLs: {len(train_dataset)}")
            print(f"[+] Val KOLs: {len(val_dataset)}")

            # Create model
            input_dim = 7  # From data_loader
            model = create_kol_predictor(input_dim, model_type='single')

            # Create trainer
            trainer = KOLPredictorTrainer(
                model,
                learning_rate=learning_rate
            )

            # Train
            save_path = f'{self.models_dir}/kol_predictor_best.pth'
            stats = trainer.train(
                train_loader,
                val_loader,
                epochs=epochs,
                save_path=save_path
            )

            # Register model
            version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.registry.register_model(
                'kol_predictor',
                version,
                metrics=stats['val_metrics'][-1],
                metadata={
                    'train_kols': len(train_dataset),
                    'val_kols': len(val_dataset),
                    'epochs_trained': stats['best_epoch'],
                    'best_metric': stats['best_metric']
                },
                model_path=save_path
            )

            print(f"\n[+] KOL predictor training complete")
            print(f"    Best RMSE: {stats['best_metric']:.4f}")

            return stats

        except Exception as e:
            print(f"[!] Error training KOL predictor: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def train_all_models(self, **kwargs) -> Dict:
        """
        Train all models

        Args:
            **kwargs: Arguments passed to training functions

        Returns:
            Dictionary with training results
        """
        print("\n" + "=" * 70)
        print(f"Starting Model Training - {datetime.now()}")
        print("=" * 70)

        results = {}

        # Train token predictor
        try:
            token_stats = self.train_token_model(**kwargs)
            results['token_predictor'] = token_stats
        except Exception as e:
            print(f"[!] Failed to train token predictor: {e}")
            results['token_predictor'] = None

        # Train KOL predictor
        try:
            kol_stats = self.train_kol_model(**kwargs)
            results['kol_predictor'] = kol_stats
        except Exception as e:
            print(f"[!] Failed to train KOL predictor: {e}")
            results['kol_predictor'] = None

        # Update last train time
        self.last_train_time = datetime.now()
        self._save_state()

        # Print summary
        print("\n" + "=" * 70)
        print("Training Summary")
        print("=" * 70)

        for model_name, stats in results.items():
            if stats and stats.get('best_metric'):
                print(f"{model_name}: {stats['best_metric']:.4f} (epoch {stats['best_epoch']})")
            else:
                print(f"{model_name}: Failed")

        return results

    def run_once(self, **kwargs):
        """Run training once"""
        if self.should_retrain():
            self.train_all_models(**kwargs)
        else:
            time_until = self.retrain_interval - (datetime.now() - self.last_train_time).total_seconds()
            hours_until = time_until / 3600
            print(f"[*] Next training in {hours_until:.1f} hours")

    def run_continuous(self, check_interval_seconds: int = 300):
        """
        Run continuous training loop

        Args:
            check_interval_seconds: How often to check (default 5 min)
        """
        self.running = True

        print(f"\n[*] Starting continuous training")
        print(f"    Retrain interval: {self.retrain_interval / 3600:.1f} hours")
        print(f"    Check interval: {check_interval_seconds} seconds")
        print(f"    Press Ctrl+C to stop\n")

        try:
            while self.running:
                # Check if should retrain
                if self.should_retrain():
                    self.train_all_models()

                # Wait before next check
                time.sleep(check_interval_seconds)

        except KeyboardInterrupt:
            print("\n[!] Training stopped by user")
            self.running = False

    def stop(self):
        """Stop continuous training"""
        self.running = False


if __name__ == "__main__":
    import os
    import torch

    # Test trainer
    print("=" * 70)
    print("Testing Continuous Trainer")
    print("=" * 70)

    trainer = ContinuousTrainer(retrain_interval_hours=6)

    # Run once
    print("\n[*] Running training once...")
    results = trainer.run_once(epochs=20, batch_size=16)

    print("\n[+] Test complete")
