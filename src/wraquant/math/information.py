"""Information-theoretic measures for financial analysis."""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import ArrayLike

from wraquant.core._coerce import coerce_array

__all__ = [
    "fisher_information",
    "mutual_information",
    "transfer_entropy",
    "entropy",
    "kl_divergence",
    "conditional_entropy",
]


def fisher_information(
    log_likelihood_fn: Callable[..., float],
    params: ArrayLike,
    dx: float = 1e-5,
) -> np.ndarray:
    """Numerical Fisher information matrix via second derivatives.

    Computes the negative of the Hessian of *log_likelihood_fn* evaluated
    at *params* using central finite differences.

    Parameters
    ----------
    log_likelihood_fn : callable
        Function ``f(params) -> float`` returning the log-likelihood.
    params : array_like
        Parameter vector at which to evaluate the information matrix.
    dx : float, optional
        Finite-difference step size (default 1e-5).

    Returns
    -------
    np.ndarray
        Fisher information matrix of shape ``(len(params), len(params))``.
        Larger diagonal entries indicate parameters that are more precisely
        estimable from the data.

    Example
    -------
    >>> import numpy as np
    >>> from wraquant.math.information import fisher_information
    >>> # Log-likelihood of Normal(mu, sigma=1): -0.5 * sum((x - mu)^2)
    >>> data = np.array([1.0, 2.0, 3.0])
    >>> ll_fn = lambda p: -0.5 * np.sum((data - p[0])**2)
    >>> fim = fisher_information(ll_fn, np.array([2.0]))
    >>> fim.shape
    (1, 1)
    >>> fim[0, 0] > 0  # positive definite
    True

    See Also
    --------
    wraquant.math.numerical.finite_difference_hessian : General Hessian computation.
    """
    params = coerce_array(params, name="params")
    n = len(params)
    fim = np.empty((n, n), dtype=float)

    for i in range(n):
        for j in range(i, n):
            # Central difference for second partial derivative
            p_pp = params.copy()
            p_pm = params.copy()
            p_mp = params.copy()
            p_mm = params.copy()

            p_pp[i] += dx
            p_pp[j] += dx

            p_pm[i] += dx
            p_pm[j] -= dx

            p_mp[i] -= dx
            p_mp[j] += dx

            p_mm[i] -= dx
            p_mm[j] -= dx

            d2 = (
                log_likelihood_fn(p_pp)
                - log_likelihood_fn(p_pm)
                - log_likelihood_fn(p_mp)
                + log_likelihood_fn(p_mm)
            ) / (4.0 * dx * dx)

            # Fisher information = negative Hessian of log-likelihood
            fim[i, j] = -d2
            fim[j, i] = -d2

    return fim


def entropy(
    data: ArrayLike,
    bins: int = 20,
    method: str = "histogram",
) -> float:
    """Shannon entropy of a data series.

    Parameters
    ----------
    data : array_like
        Input data (1-D).
    bins : int, optional
        Number of histogram bins (default 20).
    method : {'histogram'}, optional
        Discretisation method (default ``'histogram'``).

    Returns
    -------
    float
        Shannon entropy in nats.  Higher values indicate more
        dispersed (uncertain) distributions; lower values indicate
        concentrated distributions.

    Raises
    ------
    ValueError
        If *method* is not recognised.

    Example
    -------
    >>> import numpy as np
    >>> from wraquant.math.information import entropy
    >>> uniform = np.random.uniform(size=1000)
    >>> peaked = np.random.normal(0, 0.01, size=1000)
    >>> entropy(uniform) > entropy(peaked)
    True

    See Also
    --------
    conditional_entropy : Entropy of X given Y.
    mutual_information : Shared information between two variables.
    wraquant.math.spectral.spectral_entropy : Entropy of the power spectrum.
    """
    if method != "histogram":
        raise ValueError(f"Unknown method {method!r}; only 'histogram' is supported.")

    data = coerce_array(data, name="data")
    counts, _ = np.histogram(data, bins=bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def mutual_information(
    x: ArrayLike,
    y: ArrayLike,
    bins: int = 20,
) -> float:
    """Mutual information between two series (discretised).

    .. math::

        I(X; Y) = H(X) + H(Y) - H(X, Y)

    Parameters
    ----------
    x : array_like
        First data series.
    y : array_like
        Second data series.
    bins : int, optional
        Number of histogram bins per dimension (default 20).

    Returns
    -------
    float
        Mutual information in nats (>= 0).  Zero indicates independence;
        higher values indicate stronger dependence (linear or nonlinear).

    Example
    -------
    >>> import numpy as np
    >>> from wraquant.math.information import mutual_information
    >>> rng = np.random.default_rng(42)
    >>> x = rng.standard_normal(1000)
    >>> y = x + rng.standard_normal(1000) * 0.1  # highly dependent
    >>> z = rng.standard_normal(1000)             # independent
    >>> mutual_information(x, y) > mutual_information(x, z)
    True

    See Also
    --------
    transfer_entropy : Directed (causal) information flow.
    conditional_entropy : H(X | Y) = H(X) - I(X; Y).
    """
    x = coerce_array(x, name="x")
    y = coerce_array(y, name="y")

    # Joint histogram
    joint, _, _ = np.histogram2d(x, y, bins=bins)
    joint_prob = joint / joint.sum()

    # Marginals
    px = joint_prob.sum(axis=1)
    py = joint_prob.sum(axis=0)

    # MI = sum p(x,y) * log(p(x,y) / (p(x)*p(y)))
    mi = 0.0
    for i in range(len(px)):
        for j in range(len(py)):
            if joint_prob[i, j] > 0 and px[i] > 0 and py[j] > 0:
                mi += joint_prob[i, j] * np.log(joint_prob[i, j] / (px[i] * py[j]))

    return float(max(mi, 0.0))


def transfer_entropy(
    source: ArrayLike,
    target: ArrayLike,
    lag: int = 1,
    bins: int = 10,
) -> float:
    r"""Transfer entropy from *source* to *target*.

    Measures the directional information flow from *source* to *target*
    beyond what *target*'s own past explains.

    .. math::

        TE_{X \\to Y} = H(Y_t | Y_{t-k}) - H(Y_t | Y_{t-k}, X_{t-k})

    Parameters
    ----------
    source : array_like
        Source time series.
    target : array_like
        Target time series.
    lag : int, optional
        Lag order (default 1).
    bins : int, optional
        Number of histogram bins for discretisation (default 10).

    Returns
    -------
    float
        Transfer entropy in nats (>= 0).  Higher values indicate
        stronger directional information flow from *source* to *target*.

    Example
    -------
    >>> import numpy as np
    >>> from wraquant.math.information import transfer_entropy
    >>> rng = np.random.default_rng(42)
    >>> x = rng.standard_normal(500)
    >>> y = np.concatenate([[0], x[:-1]]) + rng.standard_normal(500) * 0.1
    >>> te_x_to_y = transfer_entropy(x, y, lag=1)
    >>> te_y_to_x = transfer_entropy(y, x, lag=1)
    >>> te_x_to_y > te_y_to_x  # x drives y, not vice versa
    True

    Notes
    -----
    Reference: Schreiber, T. (2000). "Measuring Information Transfer."
    *Physical Review Letters*, 85(2), 461-464.

    See Also
    --------
    mutual_information : Symmetric (undirected) dependence measure.
    wraquant.math.network.granger_network : Linear Granger causality.
    """
    source = coerce_array(source, name="source")
    target = coerce_array(target, name="target")

    n = min(len(source), len(target))
    # Align: target_future, target_past, source_past
    target_future = target[lag:n]
    target_past = target[: n - lag]
    source_past = source[: n - lag]

    # Discretise
    def _digitize(arr: np.ndarray) -> np.ndarray:
        edges = np.linspace(arr.min() - 1e-12, arr.max() + 1e-12, bins + 1)
        return np.digitize(arr, edges[1:-1])

    tf = _digitize(target_future)
    tp = _digitize(target_past)
    sp = _digitize(source_past)

    # H(target_future | target_past) - H(target_future | target_past, source_past)
    # = H(tf, tp) - H(tp) - H(tf, tp, sp) + H(tp, sp)

    def _h(*arrays: np.ndarray) -> float:
        """Joint entropy of integer-labelled arrays."""
        combined = np.column_stack(arrays)
        _, counts = np.unique(combined, axis=0, return_counts=True)
        probs = counts / counts.sum()
        return float(-np.sum(probs * np.log(probs)))

    te = _h(tf, tp) - _h(tp) - _h(tf, tp, sp) + _h(tp, sp)
    return float(max(te, 0.0))


def kl_divergence(
    p: ArrayLike,
    q: ArrayLike,
    bins: int = 20,
) -> float:
    """KL divergence D_KL(P || Q) estimated from samples.

    Parameters
    ----------
    p : array_like
        Samples from distribution P.
    q : array_like
        Samples from distribution Q.
    bins : int, optional
        Number of histogram bins (default 20).

    Returns
    -------
    float
        KL divergence in nats (>= 0).  Zero when P and Q are identical;
        larger values indicate greater distributional difference.
        Note: KL divergence is asymmetric -- D_KL(P||Q) != D_KL(Q||P).

    Example
    -------
    >>> import numpy as np
    >>> from wraquant.math.information import kl_divergence
    >>> rng = np.random.default_rng(42)
    >>> p = rng.normal(0, 1, size=5000)
    >>> q = rng.normal(0.5, 1, size=5000)
    >>> kl_divergence(p, q) > 0
    True

    See Also
    --------
    entropy : Shannon entropy of a single distribution.
    mutual_information : Symmetric dependence measure.
    """
    p_arr = coerce_array(p, name="p")
    q_arr = coerce_array(q, name="q")

    # Shared bin edges covering both distributions
    lo = min(p_arr.min(), q_arr.min())
    hi = max(p_arr.max(), q_arr.max())
    edges = np.linspace(lo, hi, bins + 1)

    p_counts, _ = np.histogram(p_arr, bins=edges)
    q_counts, _ = np.histogram(q_arr, bins=edges)

    # Convert to probabilities, add small epsilon for numerical stability
    eps = 1e-12
    p_prob = p_counts / p_counts.sum() + eps
    q_prob = q_counts / q_counts.sum() + eps

    # Re-normalise after epsilon adjustment
    p_prob = p_prob / p_prob.sum()
    q_prob = q_prob / q_prob.sum()

    kl = float(np.sum(p_prob * np.log(p_prob / q_prob)))
    return max(kl, 0.0)


def conditional_entropy(
    x: ArrayLike,
    y: ArrayLike,
    bins: int = 20,
) -> float:
    """Conditional entropy H(X | Y).

    .. math::

        H(X | Y) = H(X, Y) - H(Y)

    Parameters
    ----------
    x : array_like
        First data series.
    y : array_like
        Second data series (the conditioning variable).
    bins : int, optional
        Number of histogram bins per dimension (default 20).

    Returns
    -------
    float
        Conditional entropy in nats.  Lower values mean *Y* is more
        informative about *X*.  When H(X|Y) = 0, *X* is fully determined
        by *Y*.

    Example
    -------
    >>> import numpy as np
    >>> from wraquant.math.information import conditional_entropy
    >>> rng = np.random.default_rng(42)
    >>> x = rng.standard_normal(1000)
    >>> y = x + rng.standard_normal(1000) * 0.1  # y almost determines x
    >>> h_x_given_y = conditional_entropy(x, y)
    >>> h_x_given_y < 1.0  # low because y is informative about x
    True

    See Also
    --------
    entropy : Unconditional Shannon entropy H(X).
    mutual_information : I(X;Y) = H(X) - H(X|Y).
    """
    x = coerce_array(x, name="x")
    y = coerce_array(y, name="y")

    # Joint entropy H(X, Y)
    joint, _, _ = np.histogram2d(x, y, bins=bins)
    joint_prob = joint / joint.sum()
    joint_prob = joint_prob[joint_prob > 0]
    h_xy = float(-np.sum(joint_prob * np.log(joint_prob)))

    # Marginal entropy H(Y)
    h_y = entropy(y, bins=bins)

    return float(max(h_xy - h_y, 0.0))
