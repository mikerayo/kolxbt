"""
Token Predictor Model - PyTorch Neural Network
Predicts probability of a token reaching 3x+
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class TokenPredictor(nn.Module):
    """
    Neural network to predict if a token will reach 3x+

    Architecture:
    - Input layer (features)
    - Hidden layers with LayerNorm (works with small batches)
    - Output layer (single neuron with Sigmoid)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 32],
        dropout: float = 0.2
    ):
        """
        Initialize Token Predictor

        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
        """
        super(TokenPredictor, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout

        # Build network layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Layer normalization (works better with small batches)
            layers.append(nn.LayerNorm(hidden_dim))

            # Activation
            layers.append(nn.ReLU())

            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, 1) with probabilities
        """
        return self.network(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions (inference mode)

        Args:
            x: Input tensor

        Returns:
            Predictions as probabilities
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def get_feature_importance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Estimate feature importance using gradient-based method

        Args:
            x: Input tensor
            y: True labels

        Returns:
            Feature importance scores
        """
        self.eval()
        x.requires_grad = True

        output = self.forward(x)
        loss = F.binary_cross_entropy(output, y)

        self.zero_grad()
        loss.backward()

        # Absolute gradient magnitude as importance
        importance = torch.abs(x.grad).mean(dim=0)
        return importance


class TokenPredictorLSTM(nn.Module):
    """
    LSTM-based model for sequential token analysis
    Can analyze sequence of trades for a token
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        """
        Initialize LSTM-based predictor

        Args:
            input_dim: Number of input features per time step
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Use bidirectional LSTM
        """
        super(TokenPredictorLSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

        # Calculate output dimension
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # Fully connected layers
        self.fc1 = nn.Linear(lstm_output_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(64)
        self.layer_norm2 = nn.LayerNorm(32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            Output tensor of shape (batch_size, 1)
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)

        # Take last output
        last_out = lstm_out[:, -1, :]

        # Fully connected layers
        out = self.fc1(last_out)
        out = self.layer_norm1(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.layer_norm2(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.fc3(out)
        out = torch.sigmoid(out)

        return out


def create_token_predictor(
    input_dim: int,
    model_type: str = 'mlp',
    **kwargs
) -> nn.Module:
    """
    Factory function to create token predictor

    Args:
        input_dim: Number of input features
        model_type: Type of model ('mlp' or 'lstm')
        **kwargs: Additional arguments for model

    Returns:
        Token predictor model
    """
    if model_type == 'mlp':
        return TokenPredictor(input_dim, **kwargs)
    elif model_type == 'lstm':
        return TokenPredictorLSTM(input_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test model
    print("=" * 70)
    print("Testing Token Predictor")
    print("=" * 70)

    # Create model
    input_dim = 7  # Number of features from data_loader
    model = TokenPredictor(input_dim)

    print(f"\n[*] Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, input_dim)

    model.eval()
    with torch.no_grad():
        output = model(x)

    print(f"[+] Input shape: {x.shape}")
    print(f"[+] Output shape: {output.shape}")
    print(f"[+] Output (probabilities): {output.flatten().numpy()}")

    # Test LSTM model
    print("\n[*] Testing LSTM model...")
    lstm_model = TokenPredictorLSTM(input_dim)

    seq_len = 10
    x_seq = torch.randn(batch_size, seq_len, input_dim)

    lstm_model.eval()
    with torch.no_grad():
        output_lstm = lstm_model(x_seq)

    print(f"[+] Input shape: {x_seq.shape}")
    print(f"[+] Output shape: {output_lstm.shape}")
    print(f"[+] LSTM model parameters: {sum(p.numel() for p in lstm_model.parameters())}")
