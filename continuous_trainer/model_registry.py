"""
Model Registry for managing model versions and metadata
Tracks training history and model performance
"""

import json
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path


class ModelRegistry:
    """
    Manage model versions and training metadata
    """

    def __init__(self, registry_path: str = 'models/model_registry.json'):
        """
        Initialize model registry

        Args:
            registry_path: Path to registry JSON file
        """
        self.registry_path = registry_path
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict:
        """Load registry from file"""
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[!] Error loading registry: {e}")
                return {}
        return {}

    def _save_registry(self):
        """Save registry to file"""
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)

    def register_model(
        self,
        model_name: str,
        version: str,
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None,
        model_path: Optional[str] = None
    ):
        """
        Register a new model version

        Args:
            model_name: Name of the model (e.g., 'token_predictor')
            version: Version string (e.g., 'v1.0.0')
            metrics: Dictionary of metrics
            metadata: Additional metadata
            model_path: Path to model file
        """
        if model_name not in self.registry:
            self.registry[model_name] = {
                'name': model_name,
                'versions': [],
                'latest_version': None,
                'best_version': None
            }

        model_info = {
            'version': version,
            'trained_at': datetime.now().isoformat(),
            'metrics': metrics,
            'metadata': metadata or {},
            'model_path': model_path
        }

        # Add version
        self.registry[model_name]['versions'].append(model_info)

        # Update latest
        self.registry[model_name]['latest_version'] = version

        # Update best (based on first metric)
        if not self.registry[model_name]['best_version']:
            self.registry[model_name]['best_version'] = version
        else:
            # Get current best metrics
            best_version = self.registry[model_name]['best_version']
            best_info = self._get_version_info(model_name, best_version)
            best_metric = list(best_info['metrics'].values())[0] if best_info else 0
            new_metric = list(metrics.values())[0] if metrics else 0

            # Assume higher is better for now
            if new_metric > best_metric:
                self.registry[model_name]['best_version'] = version

        self._save_registry()
        print(f"[+] Registered {model_name} version {version}")

    def _get_version_info(self, model_name: str, version: str) -> Optional[Dict]:
        """Get info for a specific version"""
        if model_name not in self.registry:
            return None

        for v in self.registry[model_name]['versions']:
            if v['version'] == version:
                return v
        return None

    def get_best_model(self, model_name: str) -> Optional[Dict]:
        """
        Get best model information

        Args:
            model_name: Name of the model

        Returns:
            Dictionary with best model info
        """
        if model_name not in self.registry:
            return None

        best_version = self.registry[model_name].get('best_version')
        return self._get_version_info(model_name, best_version)

    def get_latest_model(self, model_name: str) -> Optional[Dict]:
        """
        Get latest model information

        Args:
            model_name: Name of the model

        Returns:
            Dictionary with latest model info
        """
        if model_name not in self.registry:
            return None

        latest_version = self.registry[model_name].get('latest_version')
        return self._get_version_info(model_name, latest_version)

    def get_model_path(self, model_name: str, version: Optional[str] = None) -> Optional[str]:
        """
        Get model file path

        Args:
            model_name: Name of the model
            version: Version to get (None for best)

        Returns:
            Path to model file
        """
        if version:
            info = self._get_version_info(model_name, version)
        else:
            info = self.get_best_model(model_name)

        return info['model_path'] if info else None

    def list_models(self) -> List[str]:
        """
        List all registered models

        Returns:
            List of model names
        """
        return list(self.registry.keys())

    def get_model_history(self, model_name: str) -> List[Dict]:
        """
        Get training history for a model

        Args:
            model_name: Name of the model

        Returns:
            List of all versions
        """
        if model_name not in self.registry:
            return []
        return self.registry[model_name]['versions']

    def compare_versions(
        self,
        model_name: str,
        metric: str = 'auc'
    ) -> List[Dict]:
        """
        Compare versions by a specific metric

        Args:
            model_name: Name of the model
            metric: Metric to compare

        Returns:
            List of versions sorted by metric
        """
        if model_name not in self.registry:
            return []

        versions = self.registry[model_name]['versions'].copy()

        # Sort by metric (descending)
        versions.sort(
            key=lambda x: x['metrics'].get(metric, 0),
            reverse=True
        )

        return versions

    def get_summary(self) -> Dict:
        """
        Get summary of all models

        Returns:
            Dictionary with model summary
        """
        summary = {}

        for model_name in self.list_models():
            best = self.get_best_model(model_name)
            latest = self.get_latest_model(model_name)

            summary[model_name] = {
                'total_versions': len(self.registry[model_name]['versions']),
                'best_version': best['version'] if best else None,
                'best_metrics': best['metrics'] if best else {},
                'latest_version': latest['version'] if latest else None,
                'latest_trained_at': latest['trained_at'] if latest else None
            }

        return summary

    def prune_old_versions(
        self,
        model_name: str,
        keep_n: int = 5
    ):
        """
        Remove old versions, keeping only the best N

        Args:
            model_name: Name of the model
            keep_n: Number of versions to keep
        """
        if model_name not in self.registry:
            return

        versions = self.registry[model_name]['versions']
        if len(versions) <= keep_n:
            return

        # Sort by training date (keep most recent)
        versions.sort(key=lambda x: x['trained_at'], reverse=True)

        # Keep best and N most recent
        best_version = self.registry[model_name]['best_version']
        to_keep = [best_version]

        for v in versions:
            if len(to_keep) >= keep_n:
                break
            if v['version'] != best_version:
                to_keep.append(v['version'])

        # Update registry
        self.registry[model_name]['versions'] = [
            v for v in versions if v['version'] in to_keep
        ]

        self._save_registry()
        print(f"[*] Pruned {model_name}: kept {len(to_keep)} versions")


def load_registry(registry_path: str = 'models/model_registry.json') -> ModelRegistry:
    """
    Load model registry

    Args:
        registry_path: Path to registry file

    Returns:
        ModelRegistry instance
    """
    return ModelRegistry(registry_path)


if __name__ == "__main__":
    # Test model registry
    print("=" * 70)
    print("Testing Model Registry")
    print("=" * 70)

    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        registry_path = os.path.join(tmpdir, 'test_registry.json')
        registry = ModelRegistry(registry_path)

        # Register some models
        print("\n[*] Registering models...")
        registry.register_model(
            'token_predictor',
            'v1.0.0',
            metrics={'auc': 0.75, 'f1': 0.70},
            metadata={'epochs': 50}
        )

        registry.register_model(
            'token_predictor',
            'v1.1.0',
            metrics={'auc': 0.80, 'f1': 0.75},
            metadata={'epochs': 100}
        )

        registry.register_model(
            'kol_predictor',
            'v1.0.0',
            metrics={'rmse': 0.15, 'r2': 0.65},
            metadata={'epochs': 75}
        )

        # Test queries
        print("\n[*] Best token predictor:")
        best = registry.get_best_model('token_predictor')
        print(f"    Version: {best['version']}")
        print(f"    Metrics: {best['metrics']}")

        print("\n[*] Latest kol predictor:")
        latest = registry.get_latest_model('kol_predictor')
        print(f"    Version: {latest['version']}")
        print(f"    Trained at: {latest['trained_at']}")

        print("\n[*] Model summary:")
        summary = registry.get_summary()
        for model_name, info in summary.items():
            print(f"\n    {model_name}:")
            print(f"      Total versions: {info['total_versions']}")
            print(f"      Best version: {info['best_version']}")
            print(f"      Best metrics: {info['best_metrics']}")

        print("\n[+] Registry test complete")
