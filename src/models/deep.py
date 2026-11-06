"""PyTorch models (via skorch): feed-forward ANN, purchase-sequence LSTM and spending autoencoder."""

from typing import Any

import numpy as np
import torch
from skorch import NeuralNetBinaryClassifier, NeuralNetRegressor
from skorch.callbacks import EarlyStopping
from skorch.dataset import Dataset
from skorch.helper import predefined_split
from torch import nn

SPEND_FEATURES = [
    "monetary_total",
    "avg_order_value",
    "avg_basket_items",
    "avg_basket_lines",
    "frequency",
    "return_rate",
    "markdown_share",
    "revenue_last_90d",
]


def to_float32(a: np.ndarray) -> np.ndarray:
    """Cast to float32 for PyTorch."""
    return np.asarray(a, dtype=np.float32)


def log1p_pos(a: np.ndarray) -> np.ndarray:
    """log(1 + x) on non-negative-clipped values (tames heavy-tailed spend)."""
    return np.log1p(np.clip(a, 0, None))


class ANN(nn.Module):
    """Two hidden layers with batch normalisation and dropout; outputs a churn logit."""

    def __init__(self, n_in: int, hidden: int = 64, dropout: float = 0.3) -> None:
        super().__init__()
        block = lambda i, o: [nn.Linear(i, o), nn.BatchNorm1d(o), nn.ReLU(), nn.Dropout(dropout)]  # noqa: E731
        self.net = nn.Sequential(*block(n_in, hidden), *block(hidden, hidden // 2), nn.Linear(hidden // 2, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class SequenceLSTM(nn.Module):
    """LSTM over a customer's order sequence; the final hidden state predicts a purchase in the horizon."""

    def __init__(self, n_in: int, hidden: int = 32, dropout: float = 0.2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.lstm(x)[1][0][-1]).squeeze(-1)


class AutoEncoder(nn.Module):
    """Symmetric bottleneck autoencoder for spending patterns."""

    def __init__(self, n_in: int, bottleneck: int = 3) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(n_in, 16), nn.ReLU(), nn.Linear(16, bottleneck))
        self.decoder = nn.Sequential(nn.Linear(bottleneck, 16), nn.ReLU(), nn.Linear(16, n_in))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def _net(cls: type, module: type, n_in: int, valid: tuple[np.ndarray, np.ndarray], cfg: dict[str, Any], **kw: Any):
    d = cfg["models"]["deep"]
    torch.manual_seed(cfg["seed"])
    return cls(
        module,
        module__n_in=n_in,
        max_epochs=d["max_epochs"],
        lr=d["lr"],
        batch_size=d["batch_size"],
        optimizer=torch.optim.Adam,
        optimizer__weight_decay=1e-4,
        iterator_train__shuffle=True,
        train_split=predefined_split(Dataset(*valid)),
        verbose=0,
        callbacks=[EarlyStopping(patience=d["patience"], load_best=True)],
        **kw,
    )


def train_classifier(
    module: type, X_tr: np.ndarray, y_tr: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, cfg: dict[str, Any]
) -> NeuralNetBinaryClassifier:
    """Fit a binary classifier (ANN or LSTM) with class-weighted BCE and early stopping on validation loss."""
    pos_weight = torch.tensor((1 - y_tr.mean()) / y_tr.mean(), dtype=torch.float32)
    y_tr, y_val = y_tr.astype(np.float32), y_val.astype(np.float32)
    net = _net(NeuralNetBinaryClassifier, module, X_tr.shape[-1], (X_val, y_val), cfg, criterion__pos_weight=pos_weight)
    return net.fit(X_tr, y_tr)


def train_autoencoder(X_tr: np.ndarray, X_val: np.ndarray, cfg: dict[str, Any]) -> NeuralNetRegressor:
    """Fit the autoencoder to reconstruct its own (scaled) input."""
    return _net(NeuralNetRegressor, AutoEncoder, X_tr.shape[1], (X_val, X_val), cfg, criterion=nn.MSELoss).fit(
        X_tr, X_tr
    )


def reconstruction_error(net: NeuralNetRegressor, X: np.ndarray) -> np.ndarray:
    """Mean squared reconstruction error per row."""
    return ((net.predict(X) - X) ** 2).mean(axis=1)


def loss_curve(net: NeuralNetBinaryClassifier | NeuralNetRegressor) -> list[dict[str, float]]:
    """Training vs validation loss per epoch (for convergence/overfitting plots)."""
    return [
        {"epoch": e, "train": float(t), "valid": float(v)}
        for e, t, v in net.history[:, ("epoch", "train_loss", "valid_loss")]
    ]
