"""
PyTorch Data Loaders for ML Training
Creates datasets for token prediction and KOL performance prediction
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
from datetime import datetime, timedelta

from core.database import db, KOL, Trade, ClosedPosition
from core.feature_engineering import KOLFeatures
from core.ml_models import DiamondHandScorer


class TokenPredictionDataset(Dataset):
    """
    Dataset for training token 3x+ prediction model

    Features:
    - KOL stats (diamond hand score, 3x+ rate, win rate, avg hold time)
    - Trade info (amount SOL, timestamp features)
    - Market context (number of DH buyers, avg DH score)

    Label:
    - is_diamond_hand (pnl_multiple >= 3.0)
    """

    def __init__(
        self,
        start_date: datetime,
        end_date: datetime,
        min_trades_per_kol: int = 5,
        session = None
    ):
        """
        Initialize dataset

        Args:
            start_date: Start date for training data
            end_date: End date for training data
            min_trades_per_kol: Minimum trades required for KOL to be included
            session: Database session (creates new if None)
        """
        self.start_date = start_date
        self.end_date = end_date
        self.min_trades = min_trades_per_kol

        # Create session if not provided
        if session is None:
            self.session = db.get_session()
            self.should_close = True
        else:
            self.session = session
            self.should_close = False

        # Load data
        self.features, self.labels, self.metadata = self._load_data()

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Load features and labels from database

        Returns:
            features: numpy array of shape (n_samples, n_features)
            labels: numpy array of shape (n_samples,)
            metadata: list of dicts with sample info
        """
        print(f"[*] Loading token prediction data from {self.start_date} to {self.end_date}")

        # Get all closed positions in date range
        positions = self.session.query(ClosedPosition).filter(
            ClosedPosition.exit_time >= self.start_date,
            ClosedPosition.exit_time <= self.end_date
        ).all()

        print(f"[+] Found {len(positions)} closed positions")

        if not positions:
            return np.array([]), np.array([]), []

        # Calculate KOL features for all KOLs with enough trades
        scorer = DiamondHandScorer()
        kol_features_map = {}

        for pos in positions:
            if pos.kol_id not in kol_features_map:
                kol = self.session.query(KOL).get(pos.kol_id)
                if kol:
                    calc = KOLFeatures(kol, self.session)
                    features = calc.calculate_all_features()

                    # Only include KOLs with enough trades
                    if features.get('total_trades', 0) >= self.min_trades:
                        # Calculate diamond hand score
                        dh_score = scorer.calculate_score(features)
                        features['diamond_hand_score'] = dh_score
                        kol_features_map[pos.kol_id] = features

        print(f"[+] {len(kol_features_map)} KOLs with >= {self.min_trades} trades")

        # Prepare features for each position
        feature_list = []
        label_list = []
        metadata_list = []

        for pos in positions:
            if pos.kol_id not in kol_features_map:
                continue

            kol_feats = kol_features_map[pos.kol_id]

            # Extract features
            features = self._extract_features(pos, kol_feats)
            label = 1.0 if pos.pnl_multiple >= 3.0 else 0.0

            feature_list.append(features)
            label_list.append(label)

            metadata_list.append({
                'kol_id': pos.kol_id,
                'token_address': pos.token_address,
                'entry_time': pos.entry_time,
                'exit_time': pos.exit_time,
                'pnl_multiple': pos.pnl_multiple,
                'hold_time_hours': pos.hold_time_hours
            })

        features_array = np.array(feature_list, dtype=np.float32)
        labels_array = np.array(label_list, dtype=np.float32)

        print(f"[+] Created dataset with {len(features_array)} samples")
        print(f"    - Positive samples (3x+): {int(labels_array.sum())}")
        print(f"    - Negative samples (<3x): {int(len(labels_array) - labels_array.sum())}")
        print(f"    - Feature dim: {features_array.shape[1] if len(features_array) > 0 else 0}")

        return features_array, labels_array, metadata_list

    def _extract_features(self, position: ClosedPosition, kol_features: Dict) -> List[float]:
        """
        Extract feature vector for a single position

        Args:
            position: ClosedPosition object
            kol_features: KOL feature dictionary

        Returns:
            List of feature values
        """
        # KOL Features (4)
        dh_score = kol_features.get('diamond_hand_score', 0) / 100.0  # Normalize to 0-1
        three_x_rate = kol_features.get('three_x_plus_rate', 0)
        win_rate = kol_features.get('win_rate', 0)
        avg_hold = kol_features.get('avg_hold_time_hours', 0) / 24.0  # Normalize by 24h

        # Trade Features (3) - Calculate amount spent (absolute value)
        if hasattr(position, 'amount_sol') and position.amount_sol is not None:
            # Use absolute value since it's the amount invested
            amount_sol = abs(position.amount_sol)
        else:
            # Calculate from pnl and multiple
            if position.pnl_multiple > 1 and position.pnl_sol > 0:
                profit = position.pnl_sol
                amount_sol = profit / (position.pnl_multiple - 1)
            elif position.pnl_multiple < 1 and position.pnl_sol < 0:
                loss = abs(position.pnl_sol)
                amount_sol = loss / (1 - position.pnl_multiple) if position.pnl_multiple < 1 else 1.0
            else:
                amount_sol = 1.0  # Default value

        # Clamp amount_sol to reasonable range and normalize
        amount_sol = max(0.01, min(amount_sol, 1000))  # Clamp between 0.01 and 1000 SOL
        amount_sol_norm = np.log1p(amount_sol) / np.log1p(100)  # Normalize to 0-1 range

        # Time features (2)
        entry_hour = position.entry_time.hour / 24.0  # 0-1
        entry_day = position.entry_time.weekday() / 7.0  # 0-1 (Monday=0)

        # Combine all features
        features = [
            dh_score,
            three_x_rate,
            win_rate,
            avg_hold,
            amount_sol_norm,
            entry_hour,
            entry_day
        ]

        return features

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample

        Returns:
            features: torch tensor of shape (n_features,)
            label: torch tensor of shape (1,)
        """
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return features, label

    def __del__(self):
        """Close session if we created it"""
        if self.should_close and hasattr(self, 'session'):
            self.session.close()


class KOLPerformanceDataset(Dataset):
    """
    Dataset for training KOL future performance prediction model

    Features:
    - Historical performance (last 30/60/90 days)
    - Trend features (recent vs historical)
    - Consistency features

    Labels:
    - Future 7d 3x+ rate
    - Future 30d 3x+ rate
    """

    def __init__(
        self,
        prediction_date: datetime,
        history_days: int = 90,
        min_trades_per_kol: int = 10,
        session = None
    ):
        """
        Initialize dataset

        Args:
            prediction_date: Date to make prediction from
            history_days: How many days of history to use as features
            min_trades_per_kol: Minimum trades required
            session: Database session (creates new if None)
        """
        self.prediction_date = prediction_date
        self.history_days = history_days
        self.min_trades = min_trades_per_kol

        # Date ranges
        self.history_start = prediction_date - timedelta(days=history_days)

        # Future periods for labels
        self.future_7d_end = prediction_date + timedelta(days=7)
        self.future_30d_end = prediction_date + timedelta(days=30)

        # Create session if not provided
        if session is None:
            self.session = db.get_session()
            self.should_close = True
        else:
            self.session = session
            self.should_close = False

        # Load data
        self.features, self.labels_7d, self.labels_30d, self.metadata = self._load_data()

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """
        Load features and labels from database

        Returns:
            features: numpy array of shape (n_kols, n_features)
            labels_7d: numpy array of future 7d 3x+ rates
            labels_30d: numpy array of future 30d 3x+ rates
            metadata: list of dicts with KOL info
        """
        print(f"[*] Loading KOL performance data")
        print(f"    Prediction date: {self.prediction_date}")
        print(f"    History: {self.history_days} days")
        print(f"    Future: 7d and 30d")

        # Get all KOLs
        kols = self.session.query(KOL).all()

        feature_list = []
        label_7d_list = []
        label_30d_list = []
        metadata_list = []

        for kol in kols:
            # Calculate historical features
            hist_features = self._calculate_historical_features(kol)

            # Skip if not enough historical data
            if hist_features['hist_total_trades'] < self.min_trades:
                continue

            # Calculate future labels
            future_7d = self._calculate_future_performance(kol, self.prediction_date, self.future_7d_end)
            future_30d = self._calculate_future_performance(kol, self.prediction_date, self.future_30d_end)

            # Only include if we have future data
            if future_7d['total_trades'] == 0 or future_30d['total_trades'] == 0:
                continue

            # Extract feature vector
            features = self._extract_kol_features(hist_features)

            feature_list.append(features)
            label_7d_list.append(future_7d['three_x_plus_rate'])
            label_30d_list.append(future_30d['three_x_plus_rate'])

            metadata_list.append({
                'kol_id': kol.id,
                'kol_name': kol.name,
                'hist_trades': hist_features['hist_total_trades'],
                'future_7d_trades': future_7d['total_trades'],
                'future_30d_trades': future_30d['total_trades']
            })

        features_array = np.array(feature_list, dtype=np.float32) if feature_list else np.array([])
        labels_7d_array = np.array(label_7d_list, dtype=np.float32) if label_7d_list else np.array([])
        labels_30d_array = np.array(label_30d_list, dtype=np.float32) if label_30d_list else np.array([])

        print(f"[+] Created dataset with {len(features_array)} KOLs")

        return features_array, labels_7d_array, labels_30d_array, metadata_list

    def _calculate_historical_features(self, kol: KOL) -> Dict:
        """Calculate historical performance features for a KOL"""
        positions = self.session.query(ClosedPosition).filter(
            ClosedPosition.kol_id == kol.id,
            ClosedPosition.exit_time >= self.history_start,
            ClosedPosition.exit_time < self.prediction_date
        ).all()

        if not positions:
            return {
                'hist_total_trades': 0,
                'hist_win_rate': 0,
                'hist_three_x_rate': 0,
                'hist_avg_hold': 0,
                'hist_avg_multiple': 0,
                'hist_total_pnl': 0,
                'hist_consistency': 0
            }

        # Calculate metrics
        total_trades = len(positions)
        winning_trades = sum(1 for p in positions if p.is_profitable)
        three_x_trades = sum(1 for p in positions if p.pnl_multiple >= 3.0)

        hold_times = [p.hold_time_hours for p in positions]
        multiples = [p.pnl_multiple for p in positions]
        pnls = [p.pnl_sol for p in positions]

        return {
            'hist_total_trades': total_trades,
            'hist_win_rate': winning_trades / total_trades if total_trades > 0 else 0,
            'hist_three_x_rate': three_x_trades / total_trades if total_trades > 0 else 0,
            'hist_avg_hold': np.mean(hold_times) if hold_times else 0,
            'hist_avg_multiple': np.mean(multiples) if multiples else 0,
            'hist_total_pnl': sum(pnls),
            'hist_consistency': 1.0 / (1.0 + np.std(multiples) / (abs(np.mean(multiples)) + 1e-6)) if len(multiples) > 1 else 0
        }

    def _calculate_future_performance(self, kol: KOL, start_date: datetime, end_date: datetime) -> Dict:
        """Calculate future performance for a KOL"""
        positions = self.session.query(ClosedPosition).filter(
            ClosedPosition.kol_id == kol.id,
            ClosedPosition.exit_time >= start_date,
            ClosedPosition.exit_time < end_date
        ).all()

        if not positions:
            return {
                'total_trades': 0,
                'three_x_plus_rate': 0
            }

        total_trades = len(positions)
        three_x_trades = sum(1 for p in positions if p.pnl_multiple >= 3.0)

        return {
            'total_trades': total_trades,
            'three_x_plus_rate': three_x_trades / total_trades if total_trades > 0 else 0
        }

    def _extract_kol_features(self, hist_features: Dict) -> List[float]:
        """Extract feature vector from historical features"""
        return [
            hist_features['hist_total_trades'] / 100.0,  # Normalize by 100 trades
            hist_features['hist_win_rate'],
            hist_features['hist_three_x_rate'],
            hist_features['hist_avg_hold'] / 24.0,  # Normalize by 24h
            hist_features['hist_avg_multiple'] / 5.0,  # Normalize by 5x
            np.log1p(abs(hist_features['hist_total_pnl'])) / 10.0,  # Log normalize PnL
            hist_features['hist_consistency']
        ]

    def __len__(self) -> int:
        return len(self.labels_7d)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample

        Returns:
            features: torch tensor of shape (n_features,)
            label_7d: torch tensor with future 7d 3x+ rate
            label_30d: torch tensor with future 30d 3x+ rate
        """
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label_7d = torch.tensor(self.labels_7d[idx], dtype=torch.float32)
        label_30d = torch.tensor(self.labels_30d[idx], dtype=torch.float32)
        return features, label_7d, label_30d

    def __del__(self):
        """Close session if we created it"""
        if self.should_close and hasattr(self, 'session'):
            self.session.close()


def create_data_loaders(
    dataset_class,
    train_params: Dict,
    val_params: Dict,
    batch_size: int = 32,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders

    Args:
        dataset_class: Dataset class to instantiate
        train_params: Parameters for training dataset
        val_params: Parameters for validation dataset
        batch_size: Batch size
        num_workers: Number of worker processes

    Returns:
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
    """
    train_dataset = dataset_class(**train_params)
    val_dataset = dataset_class(**val_params)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Test datasets
    print("=" * 70)
    print("Testing ML Datasets")
    print("=" * 70)

    from datetime import datetime, timedelta

    # Test TokenPredictionDataset
    print("\n[*] Testing TokenPredictionDataset...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    token_dataset = TokenPredictionDataset(start_date, end_date)
    if len(token_dataset) > 0:
        features, label = token_dataset[0]
        print(f"[+] Sample features shape: {features.shape}")
        print(f"[+] Sample label: {label.item()}")
        print(f"[+] Dataset size: {len(token_dataset)}")
    else:
        print("[!] No data available - need more closed positions")

    # Test KOLPerformanceDataset
    print("\n[*] Testing KOLPerformanceDataset...")
    prediction_date = datetime.now() - timedelta(days=35)  # Predict from 35 days ago

    kol_dataset = KOLPerformanceDataset(prediction_date)
    if len(kol_dataset) > 0:
        features, label_7d, label_30d = kol_dataset[0]
        print(f"[+] Sample features shape: {features.shape}")
        print(f"[+] Sample 7d label: {label_7d.item():.3f}")
        print(f"[+] Sample 30d label: {label_30d.item():.3f}")
        print(f"[+] Dataset size: {len(kol_dataset)}")
    else:
        print("[!] No data available - need more historical data")
