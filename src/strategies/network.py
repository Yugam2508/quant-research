"""
network.py — signal-to-weights neural network using Flax

Architecture:
  input  : (n_signals,)  — momentum, vol, RSI, z-score per asset
  hidden : 64 → 32 units with ReLU
  output : (n_assets,) raw logits → softmax → portfolio weights

The whole thing is differentiable end-to-end, so gradients from the
Sharpe loss flow all the way back through the network weights.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence


class SignalNetwork(nn.Module):
    """
    Maps a flat signal vector to portfolio weights.

    Attributes:
        n_assets    : number of assets in the portfolio
        hidden_dims : sizes of hidden layers
        dropout_rate: dropout for regularisation (0 = off)
    """
    n_assets:     int
    hidden_dims:  Sequence[int] = (64, 32)
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """
        Forward pass.

        Args:
            x        : (n_signals,) input signal vector
            training : if True, applies dropout

        Returns:
            (n_assets,) portfolio weights summing to 1
        """
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
            x = nn.LayerNorm()(x)           # stabilises training
            if self.dropout_rate > 0:
                x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)

        # output layer: raw logits
        logits = nn.Dense(self.n_assets)(x)

        # softmax → valid long-only weights (all > 0, sum to 1)
        weights = jax.nn.softmax(logits)
        return weights


class LinearSignalModel(nn.Module):
    """
    Simpler baseline: single linear layer, no hidden units.
    Good for comparison — if the deep network doesn't beat this,
    the extra complexity isn't justified.
    """
    n_assets: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        logits = nn.Dense(self.n_assets)(x)
        return jax.nn.softmax(logits)


def build_signal_matrix(snapshot_history: list) -> jnp.ndarray:
    """
    Convert a list of daily snapshots into a flat signal vector.

    Each snapshot is a dict: {ticker: {ret_1d, ret_5d, vol_20d, rsi_14, cs_zscore}}
    We concatenate all signals across all tickers into one flat vector.

    Returns:
        (n_tickers * n_signals_per_ticker,) normalised signal vector
    """
    import numpy as np

    signal_keys = ["ret_1d", "ret_5d", "ret_20d", "vol_20d", "rsi_14", "cs_zscore"]
    row = []
    for ticker_data in snapshot_history:
        for key in signal_keys:
            val = ticker_data.get(key, 0.0)
            row.append(0.0 if (val is None or np.isnan(float(val))) else float(val))

    arr = np.array(row, dtype=np.float32)
    # z-score normalise the input
    std = arr.std() + 1e-6
    return jnp.array((arr - arr.mean()) / std)


# ── sanity check ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np

    n_assets  = 10
    n_signals = n_assets * 6   # 6 signals per asset

    model = SignalNetwork(n_assets=n_assets)
    key   = jax.random.PRNGKey(0)
    x     = jnp.ones(n_signals)

    # initialise parameters
    params = model.init(key, x)
    weights = model.apply(params, x)

    print(f"input shape   : {x.shape}")
    print(f"output shape  : {weights.shape}")
    print(f"weights sum   : {weights.sum():.6f}  (should be 1.0)")
    print(f"weights       : {weights}")
    print(f"\nnetwork initialised correctly ✓")
