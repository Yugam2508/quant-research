"""
sharpe.py — differentiable portfolio objectives in JAX

The key insight: Sharpe ratio is just mean/std of portfolio returns.
Both are differentiable operations, so JAX can compute
d(Sharpe)/d(weights) automatically via reverse-mode autodiff.

We *maximise* Sharpe by *minimising* its negative — standard
gradient descent convention.
"""

import jax
import jax.numpy as jnp


# ── core loss ─────────────────────────────────────────────────────────────────

def portfolio_returns(weights: jnp.ndarray, asset_returns: jnp.ndarray) -> jnp.ndarray:
    """
    Compute daily portfolio returns.

    Args:
        weights      : (n_assets,)  portfolio weights, should sum to 1
        asset_returns: (n_days, n_assets)  daily asset returns

    Returns:
        (n_days,) daily portfolio returns
    """
    return asset_returns @ weights   # matrix multiply: each day's return is w·r


def sharpe_ratio(weights: jnp.ndarray, asset_returns: jnp.ndarray,
                 eps: float = 1e-6) -> jnp.ndarray:
    """
    Annualised Sharpe ratio of a portfolio.

    Sharpe = mean(r_p) / std(r_p) * sqrt(252)

    The eps prevents division by zero when std ≈ 0.
    """
    r_p = portfolio_returns(weights, asset_returns)
    return jnp.mean(r_p) / (jnp.std(r_p) + eps) * jnp.sqrt(252.0)


def sharpe_loss(weights: jnp.ndarray, asset_returns: jnp.ndarray,
                tc_bps: float = 10.0) -> jnp.ndarray:
    """
    Loss = -Sharpe + transaction cost penalty.

    Transaction cost: penalises weights that deviate far from equal weight.
    tc_bps: round-trip cost in basis points (10bps = 0.10% = realistic ETF cost).

    We minimise this — so maximising Sharpe while penalising turnover.
    """
    r_p = portfolio_returns(weights, asset_returns)

    # Sharpe term
    sr = jnp.mean(r_p) / (jnp.std(r_p) + 1e-6) * jnp.sqrt(252.0)

    # transaction cost: L2 deviation from equal weight (proxy for turnover)
    n = weights.shape[0]
    equal_w = jnp.ones(n) / n
    tc_penalty = tc_bps / 10000.0 * jnp.sum((weights - equal_w) ** 2) * 252.0

    return -sr + tc_penalty


# ── portfolio constraints ──────────────────────────────────────────────────────

def long_only_weights(raw: jnp.ndarray) -> jnp.ndarray:
    """
    Map raw unconstrained values → valid long-only portfolio weights.
    Softmax ensures: all weights > 0, sum to 1.
    """
    return jax.nn.softmax(raw)


def long_short_weights(raw: jnp.ndarray) -> jnp.ndarray:
    """
    Map raw values → long/short weights that sum to 0.
    Top half long, bottom half short, dollar-neutral.
    """
    n = raw.shape[0]
    half = n // 2
    normalised = raw / (jnp.std(raw) + 1e-6)
    # long the top half, short the bottom half (by magnitude)
    sorted_idx = jnp.argsort(normalised)[::-1]
    weights = jnp.zeros(n)
    weights = weights.at[sorted_idx[:half]].set(1.0 / half)
    weights = weights.at[sorted_idx[half:]].set(-1.0 / half)
    return weights


# ── gradient functions — this is the autodiff magic ───────────────────────────

# jax.grad returns a function that computes the gradient w.r.t. the first arg
grad_sharpe_loss = jax.jit(jax.grad(sharpe_loss, argnums=0))

# jax.value_and_grad returns (loss_value, gradient) in one pass — more efficient
value_and_grad    = jax.jit(jax.value_and_grad(sharpe_loss, argnums=0))


# ── sanity check ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np
    np.random.seed(42)

    n_days, n_assets = 252, 5
    returns = jnp.array(np.random.normal(0.0005, 0.01, (n_days, n_assets)))
    weights = long_only_weights(jnp.ones(n_assets))

    sr = sharpe_ratio(weights, returns)
    loss, grads = value_and_grad(weights, returns)

    print(f"equal weight sharpe : {sr:.3f}")
    print(f"loss                : {loss:.3f}")
    print(f"gradients           : {grads}")
    print(f"gradient shape      : {grads.shape}")
    print("\nautodiff working correctly ✓")
