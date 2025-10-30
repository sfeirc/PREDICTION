"""
Sequence models: LSTM and Transformer.
"""

import math
import torch
import torch.nn as nn
from typing import Dict


class LSTMModel(nn.Module):
    """LSTM model for sequence classification."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        # Output layer
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2),  # Binary classification
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch_size, seq_len, input_size)

        Returns:
            (batch_size, 2) logits
        """
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Take the last hidden state
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            hidden = h_n[-1]

        # Fully connected
        out = self.fc(hidden)

        return out


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)

        Returns:
            (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """Transformer model for sequence classification."""

    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_size = input_size
        self.d_model = d_model

        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Output layer
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),  # Binary classification
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch_size, seq_len, input_size)

        Returns:
            (batch_size, 2) logits
        """
        # Project to d_model
        x = self.input_projection(x)  # (batch, seq_len, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoder
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)

        # Global average pooling over sequence
        x = x.mean(dim=1)  # (batch, d_model)

        # Classification head
        out = self.fc(x)  # (batch, 2)

        return out


def create_model(model_type: str, input_size: int, config: Dict) -> nn.Module:
    """
    Create a model based on type.

    Args:
        model_type: 'lstm' or 'transformer'
        input_size: Number of input features
        config: Configuration dictionary

    Returns:
        PyTorch model
    """
    if model_type == "lstm":
        model = LSTMModel(
            input_size=input_size,
            hidden_size=config["models"]["lstm"]["hidden_size"],
            num_layers=config["models"]["lstm"]["num_layers"],
            dropout=config["models"]["lstm"]["dropout"],
            bidirectional=config["models"]["lstm"]["bidirectional"],
        )
    elif model_type == "transformer":
        model = TransformerModel(
            input_size=input_size,
            d_model=config["models"]["transformer"]["d_model"],
            nhead=config["models"]["transformer"]["nhead"],
            num_layers=config["models"]["transformer"]["num_layers"],
            dim_feedforward=config["models"]["transformer"]["dim_feedforward"],
            dropout=config["models"]["transformer"]["dropout"],
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

