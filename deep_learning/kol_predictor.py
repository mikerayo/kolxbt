"""
KOL Performance Predictor Model - PyTorch Neural Network
Predicts future 7-day and 30-day performance of KOLs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class KOLPredictor(nn.Module):
    """
    Neural network to predict KOL future performance

    Predicts:
    - Future 7-day 3x+ rate
    - Future 30-day 3x+ rate

    Architecture:
    - Feature embedding
    - Multi-head attention for feature importance
    - Separate prediction heads for 7d and 30d
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 256,
        num_heads: int = 4,
        hidden_dims: list = [128, 64],
        dropout: float = 0.4,
        use_attention: bool = True
    ):
        """
        Initialize KOL Predictor

        Args:
            input_dim: Number of input features
            embed_dim: Dimension of feature embedding
            num_heads: Number of attention heads
            hidden_dims: Hidden layer dimensions for prediction heads
            dropout: Dropout probability
            use_attention: Whether to use attention mechanism
        """
        super(KOLPredictor, self).__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.use_attention = use_attention

        # Feature embedding
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Attention mechanism
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(embed_dim)

        # Shared feature processing
        self.shared_layers = nn.Sequential(
            nn.Linear(embed_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 7-day prediction head
        self.pred_7d = nn.Sequential(
            nn.Linear(hidden_dims[1], 32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output 0-1 probability
        )

        # 30-day prediction head
        self.pred_30d = nn.Sequential(
            nn.Linear(hidden_dims[1], 32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output 0-1 probability
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            pred_7d: 7-day prediction of shape (batch_size, 1)
            pred_30d: 30-day prediction of shape (batch_size, 1)
        """
        # Feature embedding
        embedded = self.embedding(x)  # (batch_size, embed_dim)

        # Apply attention if enabled
        if self.use_attention:
            # Reshape for attention: (batch_size, 1, embed_dim)
            embedded_seq = embedded.unsqueeze(1)

            # Self-attention
            attended, attn_weights = self.attention(
                embedded_seq,
                embedded_seq,
                embedded_seq
            )

            # Remove sequence dimension and apply residual connection
            attended = attended.squeeze(1)
            embedded = self.attention_norm(embedded + attended)

        # Shared feature processing
        features = self.shared_layers(embedded)

        # Prediction heads
        pred_7d = self.pred_7d(features)
        pred_30d = self.pred_30d(features)

        return pred_7d, pred_30d

    def predict(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Make predictions (inference mode)

        Args:
            x: Input tensor
            return_attention: Whether to return attention weights

        Returns:
            pred_7d: 7-day predictions
            pred_30d: 30-day predictions
            attn_weights: Attention weights (if return_attention=True)
        """
        self.eval()

        with torch.no_grad():
            # Feature embedding
            embedded = self.embedding(x)

            # Apply attention if enabled
            attn_weights = None
            if self.use_attention:
                embedded_seq = embedded.unsqueeze(1)
                attended, attn_weights = self.attention(
                    embedded_seq,
                    embedded_seq,
                    embedded_seq
                )
                attended = attended.squeeze(1)
                embedded = self.attention_norm(embedded + attended)

            # Shared processing
            features = self.shared_layers(embedded)

            # Predictions
            pred_7d = self.pred_7d(features)
            pred_30d = self.pred_30d(features)

            if return_attention:
                return pred_7d, pred_30d, attn_weights
            else:
                return pred_7d, pred_30d


class KOLPredictorEnsemble(nn.Module):
    """
    Ensemble of KOL predictors for more robust predictions
    """

    def __init__(
        self,
        input_dim: int,
        n_models: int = 3,
        **model_kwargs
    ):
        """
        Initialize ensemble

        Args:
            input_dim: Number of input features
            n_models: Number of models in ensemble
            **model_kwargs: Arguments passed to each KOLPredictor
        """
        super(KOLPredictorEnsemble, self).__init__()

        self.models = nn.ModuleList([
            KOLPredictor(input_dim, **model_kwargs)
            for _ in range(n_models)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass - averages predictions from all models

        Args:
            x: Input tensor

        Returns:
            avg_pred_7d: Average 7-day prediction
            avg_pred_30d: Average 30-day prediction
        """
        pred_7d_list = []
        pred_30d_list = []

        for model in self.models:
            pred_7d, pred_30d = model(x)
            pred_7d_list.append(pred_7d)
            pred_30d_list.append(pred_30d)

        # Average predictions
        avg_pred_7d = torch.stack(pred_7d_list).mean(dim=0)
        avg_pred_30d = torch.stack(pred_30d_list).mean(dim=0)

        return avg_pred_7d, avg_pred_30d

    def predict_with_uncertainty(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Make predictions with uncertainty estimates

        Args:
            x: Input tensor

        Returns:
            mean_pred_7d: Mean 7-day prediction
            std_pred_7d: Std of 7-day predictions (uncertainty)
            mean_pred_30d: Mean 30-day prediction
            std_pred_30d: Std of 30-day predictions (uncertainty)
        """
        self.eval()

        with torch.no_grad():
            pred_7d_list = []
            pred_30d_list = []

            for model in self.models:
                pred_7d, pred_30d = model(x)
                pred_7d_list.append(pred_7d)
                pred_30d_list.append(pred_30d)

            # Stack predictions
            pred_7d_stack = torch.stack(pred_7d_list)  # (n_models, batch_size, 1)
            pred_30d_stack = torch.stack(pred_30d_list)

            # Calculate mean and std
            mean_pred_7d = pred_7d_stack.mean(dim=0)
            std_pred_7d = pred_7d_stack.std(dim=0)
            mean_pred_30d = pred_30d_stack.mean(dim=0)
            std_pred_30d = pred_30d_stack.std(dim=0)

            return mean_pred_7d, std_pred_7d, mean_pred_30d, std_pred_30d


def create_kol_predictor(
    input_dim: int,
    model_type: str = 'single',
    **kwargs
) -> nn.Module:
    """
    Factory function to create KOL predictor

    Args:
        input_dim: Number of input features
        model_type: Type of model ('single' or 'ensemble')
        **kwargs: Additional arguments for model

    Returns:
        KOL predictor model
    """
    if model_type == 'single':
        return KOLPredictor(input_dim, **kwargs)
    elif model_type == 'ensemble':
        return KOLPredictorEnsemble(input_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test model
    print("=" * 70)
    print("Testing KOL Predictor")
    print("=" * 70)

    # Create model
    input_dim = 7  # Number of features from data_loader
    model = KOLPredictor(input_dim)

    print(f"\n[*] Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, input_dim)

    model.eval()
    with torch.no_grad():
        pred_7d, pred_30d = model(x)

    print(f"[+] Input shape: {x.shape}")
    print(f"[+] 7-day prediction shape: {pred_7d.shape}")
    print(f"[+] 30-day prediction shape: {pred_30d.shape}")
    print(f"[+] 7-day predictions: {pred_7d.flatten().numpy()}")
    print(f"[+] 30-day predictions: {pred_30d.flatten().numpy()}")

    # Test ensemble
    print("\n[*] Testing ensemble model...")
    ensemble = KOLPredictorEnsemble(input_dim, n_models=3)

    print(f"[+] Ensemble parameters: {sum(p.numel() for p in ensemble.parameters())}")

    ensemble.eval()
    with torch.no_grad():
        mean_7d, std_7d, mean_30d, std_30d = ensemble.predict_with_uncertainty(x)

    print(f"[+] Mean 7d: {mean_7d.flatten().numpy()}")
    print(f"[+] Std 7d (uncertainty): {std_7d.flatten().numpy()}")
    print(f"[+] Mean 30d: {mean_30d.flatten().numpy()}")
    print(f"[+] Std 30d (uncertainty): {std_30d.flatten().numpy()}")
