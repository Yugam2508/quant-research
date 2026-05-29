"""
trainer.py — walk-forward training loop

The only honest way to backtest a learned strategy:
  1. Train on a window of past data (252 days)
  2. Test on the NEXT period (63 days) — data the model never saw
  3. Roll forward by 21 days and repeat

This gives a realistic out-of-sample performance curve.

The training loop:
  for each epoch:
    loss, grads = value_and_grad(weights_from_network, train_returns)
    params = params - lr * grads   (via optax)
"""

import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from .sharpe import sharpe_loss, sharpe_ratio, long_only_weights
from .network import SignalNetwork, LinearSignalModel

console = Console()


@dataclass
class TrainConfig:
    train_window: int   = 252    # days to train on
    test_window:  int   = 63     # days to test on (1 quarter)
    step_size:    int   = 21     # roll forward by 1 month
    n_epochs:     int   = 200    # gradient steps per window
    learning_rate:float = 1e-3
    tc_bps:       float = 10.0   # transaction cost in basis points
    hidden_dims:  tuple = (64, 32)
    dropout_rate: float = 0.1
    seed:         int   = 42


@dataclass
class WalkForwardResults:
    """Stores the full walk-forward backtest results."""
    oos_returns:     np.ndarray          # out-of-sample daily returns
    oos_dates:       list                # corresponding dates
    train_sharpes:   list                # in-sample Sharpe per window
    oos_sharpes:     list                # out-of-sample Sharpe per window
    weight_history:  list                # portfolio weights per window
    benchmark_returns: np.ndarray        # equal-weight benchmark


def _make_loss_fn(model, params, signal_matrix, tc_bps):
    """Create a loss function that maps asset_returns → scalar loss."""
    def loss_fn(asset_returns):
        raw_weights = model.apply(params, signal_matrix)
        return sharpe_loss(raw_weights, asset_returns, tc_bps=tc_bps)
    return loss_fn


def train_one_window(
    model,
    signal_matrix: jnp.ndarray,
    train_returns: jnp.ndarray,
    config: TrainConfig,
    rng_key,
) -> tuple:
    """
    Train the network on one walk-forward window.

    Returns:
        (trained_params, final_train_sharpe, weight_history)
    """
    # initialise fresh params for this window
    params = model.init(rng_key, signal_matrix)

    # optax Adam optimiser — adaptive learning rate, works well for this
    optimiser = optax.adam(config.learning_rate)
    opt_state = optimiser.init(params)

    best_params  = params
    best_sharpe  = -999.0
    sharpe_curve = []

    for epoch in range(config.n_epochs):
        # forward pass: get weights from network
        weights = model.apply(params, signal_matrix)

        # compute loss and gradients w.r.t. params (not weights directly)
        def loss_for_params(p):
            w = model.apply(p, signal_matrix)
            return sharpe_loss(w, train_returns, tc_bps=config.tc_bps)

        loss, grads = jax.value_and_grad(loss_for_params)(params)
        updates, opt_state_new = optimiser.update(grads, opt_state)
        params    = optax.apply_updates(params, updates)
        opt_state = opt_state_new

        if epoch % 20 == 0:
            w = model.apply(params, signal_matrix)
            sr = float(sharpe_ratio(w, train_returns))
            sharpe_curve.append(sr)
            if sr > best_sharpe:
                best_sharpe = sr
                best_params = params

    return best_params, best_sharpe, sharpe_curve


def walk_forward_backtest(
    prices: "pd.DataFrame",
    signal_fn,
    config: TrainConfig = None,
    model_type: str = "deep",
) -> WalkForwardResults:
    """
    Run the full walk-forward backtest.

    Args:
        prices     : DataFrame of adjusted close prices (date × ticker)
        signal_fn  : function(prices_window) → signal_matrix (jnp.ndarray)
        config     : TrainConfig
        model_type : "deep" or "linear"

    Returns:
        WalkForwardResults
    """
    if config is None:
        config = TrainConfig()

    import pandas as pd

    # compute daily returns
    returns = prices.pct_change().dropna()
    returns_np = returns.values.astype(np.float32)
    dates  = returns.index.tolist()
    n_days, n_assets = returns_np.shape

    console.print(f"[cyan]walk-forward backtest[/cyan]")
    console.print(f"  assets: {n_assets}, days: {n_days}")
    console.print(f"  train: {config.train_window}d, test: {config.test_window}d, step: {config.step_size}d")

    # build model
    # signal_matrix size = n_assets * 6 signals
    n_signals = n_assets * 6
    if model_type == "deep":
        model = SignalNetwork(n_assets=n_assets, hidden_dims=config.hidden_dims)
    else:
        model = LinearSignalModel(n_assets=n_assets)

    rng = jax.random.PRNGKey(config.seed)

    oos_returns_list  = []
    oos_dates_list    = []
    train_sharpes     = []
    oos_sharpes_list  = []
    weight_history    = []
    benchmark_list    = []

    # walk forward
    starts = range(
        config.train_window,
        n_days - config.test_window,
        config.step_size
    )
    n_windows = len(list(starts))

    with Progress(
        SpinnerColumn(), TextColumn("[cyan]{task.description}"),
        BarColumn(), TextColumn("{task.completed}/{task.total} windows"),
        console=console
    ) as progress:
        task = progress.add_task("training...", total=n_windows)

        for i, train_end in enumerate(range(
            config.train_window,
            n_days - config.test_window,
            config.step_size
        )):
            train_start = train_end - config.train_window
            test_end    = min(train_end + config.test_window, n_days)

            train_ret = jnp.array(returns_np[train_start:train_end])
            test_ret  = jnp.array(returns_np[train_end:test_end])

            # build signal matrix from training window
            signal_matrix = signal_fn(prices.iloc[train_start:train_end])

            # make sure signal_matrix is right size; pad/truncate if needed
            expected_size = n_assets * 6
            if signal_matrix.shape[0] != expected_size:
                if signal_matrix.shape[0] < expected_size:
                    signal_matrix = jnp.pad(signal_matrix, (0, expected_size - signal_matrix.shape[0]))
                else:
                    signal_matrix = signal_matrix[:expected_size]

            # train
            rng, subkey = jax.random.split(rng)
            params, train_sr, _ = train_one_window(
                model, signal_matrix, train_ret, config, subkey
            )

            # get weights for test period
            weights = np.array(model.apply(params, signal_matrix))
            oos_ret = returns_np[train_end:test_end] @ weights

            # benchmark: equal weight
            bench_ret = returns_np[train_end:test_end].mean(axis=1)

            oos_sr = float(oos_ret.mean() / (oos_ret.std() + 1e-6) * np.sqrt(252))

            oos_returns_list.append(oos_ret)
            oos_dates_list.extend(dates[train_end:test_end])
            train_sharpes.append(train_sr)
            oos_sharpes_list.append(oos_sr)
            weight_history.append({
                "date":    dates[train_end],
                "weights": weights,
                "tickers": prices.columns.tolist(),
            })
            benchmark_list.append(bench_ret)

            progress.advance(task)
            progress.update(task, description=f"window {i+1}/{n_windows} — OOS Sharpe {oos_sr:+.2f}")

    oos_returns   = np.concatenate(oos_returns_list)
    bench_returns = np.concatenate(benchmark_list)

    return WalkForwardResults(
        oos_returns=oos_returns,
        oos_dates=oos_dates_list,
        train_sharpes=train_sharpes,
        oos_sharpes=oos_sharpes_list,
        weight_history=weight_history,
        benchmark_returns=bench_returns,
    )
