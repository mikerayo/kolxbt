"""
Training Pipeline for PyTorch Models
Handles training, validation, and model saving
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional, Callable
from datetime import datetime
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, mean_squared_error, r2_score, mean_absolute_error
import json
import os

from .token_predictor import TokenPredictor
from .kol_predictor import KOLPredictor
from .data_loader import TokenPredictionDataset, KOLPerformanceDataset
from .model_utils import save_model, load_model


class TrainingStats:
    """Track training statistics"""

    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        self.best_metric = None
        self.best_epoch = 0
        self.patience_counter = 0

    def update(self, train_loss: float, val_loss: float, val_metrics: Dict):
        """Update statistics"""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.val_metrics.append(val_metrics)

    def is_best(self, current_metric: float, higher_is_better: bool = True) -> bool:
        """Check if current metric is best"""
        if self.best_metric is None:
            return True

        if higher_is_better:
            return current_metric > self.best_metric
        else:
            return current_metric < self.best_metric

    def update_best(self, metric: float, epoch: int):
        """Update best metric"""
        self.best_metric = metric
        self.best_epoch = epoch
        self.patience_counter = 0

    def increment_patience(self):
        """Increment patience counter"""
        self.patience_counter += 1


class TokenPredictorTrainer:
    """
    Trainer for Token Predictor model
    """

    def __init__(
        self,
        model: TokenPredictor,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5
    ):
        """
        Initialize trainer

        Args:
            model: TokenPredictor model
            device: Device to train on
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.stats = TrainingStats()

    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch

        Args:
            train_loader: Training data loader

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for features, labels in train_loader:
            features = features.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(features).flatten()  # Safe flatten
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches if n_batches > 0 else 0.0

    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict]:
        """
        Validate model

        Args:
            val_loader: Validation data loader

        Returns:
            Average validation loss
            Dictionary of metrics
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        n_batches = 0

        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(features).flatten()  # Safe flatten
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                all_predictions.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                n_batches += 1

        avg_loss = total_loss / n_batches if n_batches > 0 else 0.0

        # Calculate metrics
        predictions = np.array(all_predictions)
        labels = np.array(all_labels)

        # Convert to binary predictions
        pred_binary = (predictions >= 0.5).astype(int)

        # Calculate metrics
        auc = roc_auc_score(labels, predictions) if len(labels) > 0 else 0.0
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, pred_binary, average='binary', zero_division=0
        )

        metrics = {
            'auc': float(auc),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }

        return avg_loss, metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        save_path: str = 'models/token_predictor_best.pth',
        metric_to_watch: str = 'auc'
    ) -> Dict:
        """
        Full training loop

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of epochs
            early_stopping_patience: Patience for early stopping
            save_path: Path to save best model
            metric_to_watch: Metric to monitor for early stopping

        Returns:
            Training statistics
        """
        print(f"[*] Starting training for {epochs} epochs")
        print(f"    Device: {self.device}")
        print(f"    Metric to watch: {metric_to_watch} (higher is better)")

        higher_is_better = True  # For AUC

        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader)

            # Validate
            val_loss, val_metrics = self.validate(val_loader)

            # Update stats
            self.stats.update(train_loss, val_loss, val_metrics)

            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss: {val_loss:.4f}")
                print(f"  Val AUC: {val_metrics['auc']:.4f}")
                print(f"  Val F1: {val_metrics['f1']:.4f}")

            # Check if best model
            current_metric = val_metrics[metric_to_watch]
            if self.stats.is_best(current_metric, higher_is_better):
                print(f"  [*] Best {metric_to_watch}: {current_metric:.4f} - Saving model")
                self.stats.update_best(current_metric, epoch + 1)
                save_model(self.model, save_path, val_metrics)

            # Early stopping
            else:
                self.stats.increment_patience()
                if self.stats.patience_counter >= early_stopping_patience:
                    print(f"\n[+] Early stopping triggered at epoch {epoch+1}")
                    break

        print(f"\n[+] Training complete")
        print(f"    Best {metric_to_watch}: {self.stats.best_metric:.4f} at epoch {self.stats.best_epoch}")

        return {
            'train_losses': self.stats.train_losses,
            'val_losses': self.stats.val_losses,
            'val_metrics': self.stats.val_metrics,
            'best_metric': self.stats.best_metric,
            'best_epoch': self.stats.best_epoch
        }


class KOLPredictorTrainer:
    """
    Trainer for KOL Predictor model
    """

    def __init__(
        self,
        model: KOLPredictor,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5
    ):
        """
        Initialize trainer

        Args:
            model: KOLPredictor model
            device: Device to train on
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.stats = TrainingStats()

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for features, labels_7d, labels_30d in train_loader:
            features = features.to(self.device)
            labels_7d = labels_7d.to(self.device)
            labels_30d = labels_30d.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            pred_7d, pred_30d = self.model(features)

            # Combined loss (both 7d and 30d predictions)
            loss_7d = self.criterion(pred_7d.flatten(), labels_7d)
            loss_30d = self.criterion(pred_30d.flatten(), labels_30d)
            loss = (loss_7d + loss_30d) / 2.0

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches if n_batches > 0 else 0.0

    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict]:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        pred_7d_all = []
        pred_30d_all = []
        labels_7d_all = []
        labels_30d_all = []
        n_batches = 0

        with torch.no_grad():
            for features, labels_7d, labels_30d in val_loader:
                features = features.to(self.device)
                labels_7d = labels_7d.to(self.device)
                labels_30d = labels_30d.to(self.device)

                pred_7d, pred_30d = self.model(features)

                loss_7d = self.criterion(pred_7d.flatten(), labels_7d)
                loss_30d = self.criterion(pred_30d.flatten(), labels_30d)
                loss = (loss_7d + loss_30d) / 2.0

                total_loss += loss.item()
                pred_7d_all.extend(pred_7d.flatten().cpu().numpy())
                pred_30d_all.extend(pred_30d.flatten().cpu().numpy())
                labels_7d_all.extend(labels_7d.cpu().numpy())
                labels_30d_all.extend(labels_30d.cpu().numpy())
                n_batches += 1

        avg_loss = total_loss / n_batches if n_batches > 0 else 0.0

        # Calculate metrics
        rmse_7d = np.sqrt(mean_squared_error(labels_7d_all, pred_7d_all))
        rmse_30d = np.sqrt(mean_squared_error(labels_30d_all, pred_30d_all))
        mae_7d = mean_absolute_error(labels_7d_all, pred_7d_all)
        mae_30d = mean_absolute_error(labels_30d_all, pred_30d_all)
        r2_7d = r2_score(labels_7d_all, pred_7d_all)
        r2_30d = r2_score(labels_30d_all, pred_30d_all)

        metrics = {
            'rmse_7d': float(rmse_7d),
            'rmse_30d': float(rmse_30d),
            'mae_7d': float(mae_7d),
            'mae_30d': float(mae_30d),
            'r2_7d': float(r2_7d),
            'r2_30d': float(r2_30d),
            'avg_rmse': float((rmse_7d + rmse_30d) / 2)
        }

        return avg_loss, metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        save_path: str = 'models/kol_predictor_best.pth',
        metric_to_watch: str = 'avg_rmse'
    ) -> Dict:
        """Full training loop"""
        print(f"[*] Starting KOL predictor training for {epochs} epochs")
        print(f"    Device: {self.device}")
        print(f"    Metric to watch: {metric_to_watch} (lower is better)")

        higher_is_better = False  # For RMSE

        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader)

            # Validate
            val_loss, val_metrics = self.validate(val_loader)

            # Update stats
            self.stats.update(train_loss, val_loss, val_metrics)

            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss: {val_loss:.4f}")
                print(f"  Val RMSE (7d): {val_metrics['rmse_7d']:.4f}")
                print(f"  Val RMSE (30d): {val_metrics['rmse_30d']:.4f}")
                print(f"  Val R² (7d): {val_metrics['r2_7d']:.4f}")
                print(f"  Val R² (30d): {val_metrics['r2_30d']:.4f}")

            # Check if best model
            current_metric = val_metrics[metric_to_watch]
            if self.stats.is_best(current_metric, higher_is_better):
                print(f"  [*] Best {metric_to_watch}: {current_metric:.4f} - Saving model")
                self.stats.update_best(current_metric, epoch + 1)
                save_model(self.model, save_path, val_metrics)

            # Early stopping
            else:
                self.stats.increment_patience()
                if self.stats.patience_counter >= early_stopping_patience:
                    print(f"\n[+] Early stopping triggered at epoch {epoch+1}")
                    break

        print(f"\n[+] Training complete")
        print(f"    Best {metric_to_watch}: {self.stats.best_metric:.4f} at epoch {self.stats.best_epoch}")

        return {
            'train_losses': self.stats.train_losses,
            'val_losses': self.stats.val_losses,
            'val_metrics': self.stats.val_metrics,
            'best_metric': self.stats.best_metric,
            'best_epoch': self.stats.best_epoch
        }


if __name__ == "__main__":
    print("Training pipeline module - use with train_token_predictor or train_kol_predictor")
