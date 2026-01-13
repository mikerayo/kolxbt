"""
Batch Predictor for Real-time Inference
Makes predictions using trained ML models
"""

import torch
import numpy as np
import os
from typing import Dict, List, Optional, Union
from datetime import datetime

from deep_learning.token_predictor import TokenPredictor, create_token_predictor
from deep_learning.kol_predictor import KOLPredictor, create_kol_predictor
from deep_learning.model_utils import load_model
from continuous_trainer.model_registry import ModelRegistry
from database import db, KOL, Trade, ClosedPosition
from feature_engineering import KOLFeatures
from ml_models import DiamondHandScorer


class BatchPredictor:
    """
    Batch predictor for making predictions with trained models
    """

    def __init__(
        self,
        device: str = 'cpu',
        models_dir: str = 'models',
        registry_path: str = 'models/model_registry.json'
    ):
        """
        Initialize batch predictor

        Args:
            device: Device to run inference on
            models_dir: Directory containing model files
            registry_path: Path to model registry
        """
        self.device = device
        self.models_dir = models_dir
        self.registry = ModelRegistry(registry_path)

        self.token_model = None
        self.kol_model = None
        self.scaler = None

    def load_token_predictor(self, model_path: Optional[str] = None) -> TokenPredictor:
        """
        Load token predictor model

        Args:
            model_path: Path to model file (uses best from registry if None)

        Returns:
            Loaded model
        """
        if model_path is None:
            model_path = self.registry.get_model_path('token_predictor')

        if model_path is None or not os.path.exists(model_path):
            raise FileNotFoundError(f"Token predictor model not found at {model_path}")

        # Create model instance
        input_dim = 7  # Feature dimension
        model = create_token_predictor(input_dim, model_type='mlp').to(self.device)

        # Load weights
        info = load_model(model, model_path, device=self.device)
        self.token_model = model

        print(f"[+] Token predictor loaded from {model_path}")
        print(f"    Metrics: {info['metrics']}")

        return model

    def load_kol_predictor(self, model_path: Optional[str] = None) -> KOLPredictor:
        """
        Load KOL predictor model

        Args:
            model_path: Path to model file (uses best from registry if None)

        Returns:
            Loaded model
        """
        if model_path is None:
            model_path = self.registry.get_model_path('kol_predictor')

        if model_path is None or not os.path.exists(model_path):
            raise FileNotFoundError(f"KOL predictor model not found at {model_path}")

        # Create model instance
        input_dim = 7  # Feature dimension
        model = create_kol_predictor(input_dim, model_type='single').to(self.device)

        # Load weights
        info = load_model(model, model_path, device=self.device)
        self.kol_model = model

        print(f"[+] KOL predictor loaded from {model_path}")
        print(f"    Metrics: {info['metrics']}")

        return model

    def load_all_models(self):
        """Load all available models"""
        try:
            self.load_token_predictor()
        except Exception as e:
            print(f"[!] Could not load token predictor: {e}")

        try:
            self.load_kol_predictor()
        except Exception as e:
            print(f"[!] Could not load KOL predictor: {e}")

    def predict_token_3x_probability(
        self,
        kol_id: int,
        amount_sol: float,
        entry_time: Optional[datetime] = None
    ) -> float:
        """
        Predict probability of token reaching 3x+

        Args:
            kol_id: KOL ID
            amount_sol: Amount of SOL invested
            entry_time: Entry time (now if None)

        Returns:
            Probability (0-1)
        """
        if self.token_model is None:
            raise RuntimeError("Token predictor not loaded")

        session = db.get_session()

        try:
            # Get KOL
            kol = session.query(KOL).get(kol_id)
            if not kol:
                return 0.0

            # Calculate KOL features
            calc = KOLFeatures(kol, session)
            kol_feats = calc.calculate_all_features()

            # Calculate diamond hand score
            scorer = DiamondHandScorer()
            dh_score = scorer.calculate_score(kol_feats) / 100.0

            # Extract features
            dh_score_norm = dh_score
            three_x_rate = kol_feats.get('three_x_plus_rate', 0)
            win_rate = kol_feats.get('win_rate', 0)
            avg_hold = kol_feats.get('avg_hold_time_hours', 0) / 24.0

            amount_sol_norm = np.log1p(amount_sol) / 10.0

            if entry_time is None:
                entry_time = datetime.now()

            entry_hour = entry_time.hour / 24.0
            entry_day = entry_time.weekday() / 7.0

            # Create feature vector
            features = torch.tensor([
                dh_score_norm,
                three_x_rate,
                win_rate,
                avg_hold,
                amount_sol_norm,
                entry_hour,
                entry_day
            ], dtype=torch.float32).unsqueeze(0).to(self.device)

            # Predict
            self.token_model.eval()
            with torch.no_grad():
                probability = self.token_model(features).item()

            return probability

        finally:
            session.close()

    def predict_kol_future_performance(
        self,
        kol_id: int
    ) -> Dict[str, float]:
        """
        Predict KOL's future 7d and 30d performance

        Args:
            kol_id: KOL ID

        Returns:
            Dictionary with predictions
        """
        if self.kol_model is None:
            raise RuntimeError("KOL predictor not loaded")

        session = db.get_session()

        try:
            # Get KOL
            kol = session.query(KOL).get(kol_id)
            if not kol:
                return {'pred_7d': 0.0, 'pred_30d': 0.0}

            # Get historical positions (last 90 days)
            from datetime import timedelta

            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)

            positions = session.query(ClosedPosition).filter(
                ClosedPosition.kol_id == kol.id,
                ClosedPosition.exit_time >= start_date,
                ClosedPosition.exit_time < end_date
            ).all()

            if not positions or len(positions) < 5:
                # Not enough data
                return {'pred_7d': 0.0, 'pred_30d': 0.0, 'has_data': False}

            # Calculate features
            total_trades = len(positions)
            winning_trades = sum(1 for p in positions if p.is_profitable)
            three_x_trades = sum(1 for p in positions if p.pnl_multiple >= 3.0)

            hold_times = [p.hold_time_hours for p in positions]
            multiples = [p.pnl_multiple for p in positions]
            pnls = [p.pnl_sol for p in positions]

            win_rate = winning_trades / total_trades
            three_x_rate = three_x_trades / total_trades
            avg_hold = np.mean(hold_times) if hold_times else 0
            avg_multiple = np.mean(multiples) if multiples else 0
            total_pnl = sum(pnls)

            consistency = 1.0 / (1.0 + np.std(multiples) / (abs(np.mean(multiples)) + 1e-6))

            # Create feature vector
            features = torch.tensor([
                total_trades / 100.0,
                win_rate,
                three_x_rate,
                avg_hold / 24.0,
                avg_multiple / 5.0,
                np.log1p(abs(total_pnl)) / 10.0,
                consistency
            ], dtype=torch.float32).unsqueeze(0).to(self.device)

            # Predict
            self.kol_model.eval()
            with torch.no_grad():
                pred_7d, pred_30d = self.kol_model(features)

            return {
                'pred_7d': pred_7d.item(),
                'pred_30d': pred_30d.item(),
                'has_data': True
            }

        finally:
            session.close()

    def predict_batch_tokens(
        self,
        trades: List[Dict]
    ) -> List[Dict]:
        """
        Predict probability for multiple trades

        Args:
            trades: List of trade dicts with kol_id, amount_sol, entry_time

        Returns:
            List of predictions
        """
        results = []

        for trade in trades:
            try:
                probability = self.predict_token_3x_probability(
                    trade['kol_id'],
                    trade['amount_sol'],
                    trade.get('entry_time')
                )

                results.append({
                    'kol_id': trade['kol_id'],
                    'probability': probability,
                    'success': True
                })
            except Exception as e:
                results.append({
                    'kol_id': trade.get('kol_id'),
                    'probability': 0.0,
                    'success': False,
                    'error': str(e)
                })

        return results

    def get_model_info(self) -> Dict:
        """
        Get information about loaded models

        Returns:
            Dictionary with model information
        """
        info = {
            'device': self.device,
            'token_predictor': None,
            'kol_predictor': None
        }

        if self.token_model is not None:
            token_info = self.registry.get_best_model('token_predictor')
            info['token_predictor'] = token_info

        if self.kol_model is not None:
            kol_info = self.registry.get_best_model('kol_predictor')
            info['kol_predictor'] = kol_info

        return info


if __name__ == "__main__":
    import os

    print("=" * 70)
    print("Testing Batch Predictor")
    print("=" * 70)

    predictor = BatchPredictor()

    # Try to load models
    print("\n[*] Loading models...")
    predictor.load_all_models()

    # Get model info
    info = predictor.get_model_info()
    print(f"\n[+] Model info:")
    print(f"    Device: {info['device']}")
    print(f"    Token predictor: {info['token_predictor']}")
    print(f"    KOL predictor: {info['kol_predictor']}")

    # Test predictions if models loaded
    if predictor.token_model:
        print("\n[*] Testing token prediction...")
        try:
            prob = predictor.predict_token_3x_probability(kol_id=1, amount_sol=1.0)
            print(f"[+] Sample prediction: {prob:.3f}")
        except Exception as e:
            print(f"[!] Prediction failed: {e}")

    print("\n[+] Test complete")
