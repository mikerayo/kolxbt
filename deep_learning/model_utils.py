"""
Model Utilities for PyTorch Models
Handles saving, loading, and model metadata
"""

import torch
import os
import json
from typing import Dict, Any, Optional
from datetime import datetime


def save_model(
    model: torch.nn.Module,
    path: str,
    metrics: Dict[str, float],
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Save model with metadata

    Args:
        model: PyTorch model to save
        path: Path to save model
        metrics: Dictionary of metrics
        metadata: Additional metadata
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Prepare save data - only save state dict, not full model
    save_data = {
        'model_state_dict': model.state_dict(),
        'metrics': metrics,
        'metadata': metadata or {},
        'saved_at': datetime.now().isoformat()
    }

    # Save model (PyTorch format)
    torch.save(save_data, path)
    print(f"[+] Model saved to {path}")

    # Save metadata separately as JSON (for easy reading)
    metadata_path = path.replace('.pth', '_metadata.json')
    metadata_only = {
        'metrics': metrics,
        'metadata': metadata or {},
        'saved_at': datetime.now().isoformat(),
        'model_path': path
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata_only, f, indent=2)


def load_model(
    model: torch.nn.Module,
    path: str,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Load model with metadata

    Args:
        model: PyTorch model instance
        path: Path to load model from
        device: Device to load model to

    Returns:
        Dictionary with metadata (metrics, saved_at, etc.)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    # Load model
    save_data = torch.load(path, map_location=device)

    # Load state dict
    model.load_state_dict(save_data['model_state_dict'])

    print(f"[+] Model loaded from {path}")

    # Return metadata
    return {
        'metrics': save_data.get('metrics', {}),
        'metadata': save_data.get('metadata', {}),
        'saved_at': save_data.get('saved_at')
    }


def get_model_info(path: str) -> Dict[str, Any]:
    """
    Get model metadata without loading the model

    Args:
        path: Path to model file

    Returns:
        Dictionary with model information
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    # Try to load metadata JSON first
    metadata_path = path.replace('.pth', '_metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            data = json.load(f)
        return {
            'metrics': data.get('metrics', {}),
            'metadata': data.get('metadata', {}),
            'saved_at': data.get('saved_at')
        }

    # If no metadata file, load the full model
    save_data = torch.load(path, map_location='cpu')
    return {
        'metrics': save_data.get('metrics', {}),
        'metadata': save_data.get('metadata', {}),
        'saved_at': save_data.get('saved_at')
    }


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count total trainable parameters in model

    Args:
        model: PyTorch model

    Returns:
        Number of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(path: str) -> float:
    """
    Get model file size in MB

    Args:
        path: Path to model file

    Returns:
        Size in MB
    """
    if not os.path.exists(path):
        return 0.0

    size_bytes = os.path.getsize(path)
    return size_bytes / (1024 * 1024)


def list_models(directory: str = 'models') -> Dict[str, Dict]:
    """
    List all models in directory with their info

    Args:
        directory: Directory to search for models

    Returns:
        Dictionary mapping model names to their info
    """
    if not os.path.exists(directory):
        return {}

    models = {}
    for filename in os.listdir(directory):
        if filename.endswith('.pth') and not filename.endswith('_metadata.json'):
            path = os.path.join(directory, filename)
            try:
                info = get_model_info(path)
                models[filename] = {
                    'path': path,
                    'size_mb': get_model_size_mb(path),
                    **info
                }
            except Exception as e:
                print(f"[!] Error loading info for {filename}: {e}")

    return models


def compare_models(model_paths: list, metric: str = 'auc') -> Dict:
    """
    Compare multiple models by a specific metric

    Args:
        model_paths: List of model file paths
        metric: Metric to compare

    Returns:
        Dictionary with comparison results
    """
    results = {}

    for path in model_paths:
        try:
            info = get_model_info(path)
            model_name = os.path.basename(path)
            results[model_name] = {
                'path': path,
                metric: info['metrics'].get(metric, None)
            }
        except Exception as e:
            print(f"[!] Error loading {path}: {e}")

    # Sort by metric
    sorted_results = dict(
        sorted(results.items(), key=lambda x: x[1].get(metric, 0), reverse=True)
    )

    return sorted_results


class ModelCheckpoint:
    """
    Model checkpoint callback for training
    """

    def __init__(
        self,
        save_path: str,
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = True
    ):
        """
        Initialize checkpoint callback

        Args:
            save_path: Path to save model
            monitor: Metric to monitor
            mode: 'min' or 'max' for metric
            save_best_only: Only save best model
        """
        self.save_path = save_path
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_value = None
        self.epoch = 0

    def __call__(
        self,
        model: torch.nn.Module,
        current_value: float,
        metrics: Dict,
        epoch: int
    ):
        """
        Check if model should be saved

        Args:
            model: PyTorch model
            current_value: Current monitored value
            metrics: All metrics
            epoch: Current epoch
        """
        self.epoch = epoch

        should_save = False

        if self.best_value is None:
            should_save = True
        elif self.mode == 'min' and current_value < self.best_value:
            should_save = True
        elif self.mode == 'max' and current_value > self.best_value:
            should_save = True

        if should_save:
            self.best_value = current_value
            save_model(
                model,
                self.save_path,
                metrics,
                metadata={'epoch': epoch, 'monitored_value': current_value}
            )
            print(f"[*] Checkpoint saved: {self.monitor} = {current_value:.4f}")

        elif not self.save_best_only:
            # Save with epoch number
            checkpoint_path = self.save_path.replace('.pth', f'_epoch{epoch}.pth')
            save_model(
                model,
                checkpoint_path,
                metrics,
                metadata={'epoch': epoch, 'monitored_value': current_value}
            )


if __name__ == "__main__":
    # Test model utilities
    print("=" * 70)
    print("Testing Model Utilities")
    print("=" * 70)

    import tempfile
    import os

    from .token_predictor import TokenPredictor

    # Create a test model
    model = TokenPredictor(input_dim=7)
    print(f"\n[*] Test model created with {count_parameters(model)} parameters")

    # Test saving
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, 'test_model.pth')

        # Save model
        save_model(
            model,
            save_path,
            metrics={'auc': 0.85, 'f1': 0.78},
            metadata={'epochs': 100, 'batch_size': 32}
        )

        # Load model
        new_model = TokenPredictor(input_dim=7)
        info = load_model(new_model, save_path)

        print(f"[+] Model loaded successfully")
        print(f"    Metrics: {info['metrics']}")
        print(f"    Saved at: {info['saved_at']}")

        # Get model info
        model_info = get_model_info(save_path)
        print(f"\n[+] Model info:")
        print(f"    Size: {get_model_size_mb(save_path):.2f} MB")
        print(f"    Metrics: {model_info['metrics']}")
