"""General-purpose numerical methods useful for quantitative finance."""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "finite_difference_gradient",
    "finite_difference_hessian",
    "newton_raphson",
    "bisection",
    "trapezoidal_integration",
    "monte_carlo_integration",
]


def finite_difference_gradient(
    fn: Callable[[np.ndarray], float],
    x: ArrayLike,
    dx: float = 1e-7,
) -> np.ndarray:
    """Numerical gradient via central finite differences.

    Parameters
    ----------
    fn : callable
        Scalar-valued function ``f(x) -> float``.
    x : array_like
        Point at which to evaluate the gradient.
    dx : float, optional
        Step size (default 1e-7).

    Returns
    -------
    np.ndarray
        Gradient vector of the same shape as *x*.
    """
    x = np.asarray(x, dtype=float)
    grad = np.empty_like(x)

    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += dx
        x_minus[i] -= dx
        grad[i] = (fn(x_plus) - fn(x_minus)) / (2.0 * dx)

    return grad


def finite_difference_hessian(
    fn: Callable[[np.ndarray], float],
    x: ArrayLike,
    dx: float = 1e-5,
) -> np.ndarray:
    """Numerical Hessian matrix via central finite differences.

    Parameters
    ----------
    fn : callable
        Scalar-valued function ``f(x) -> float``.
    x : array_like
        Point at which to evaluate the Hessian.
    dx : float, optional
        Step size (default 1e-5).

    Returns
    -------
    np.ndarray
        Hessian matrix of shape ``(len(x), len(x))``.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    hess = np.empty((n, n), dtype=float)

    for i in range(n):
        for j in range(i, n):
            x_pp = x.copy()
            x_pm = x.copy()
            x_mp = x.copy()
            x_mm = x.copy()

            x_pp[i] += dx
            x_pp[j] += dx
            x_pm[i] += dx
            x_pm[j] -= dx
            x_mp[i] -= dx
            x_mp[j] += dx
            x_mm[i] -= dx
            x_mm[j] -= dx

            d2 = (fn(x_pp) - fn(x_pm) - fn(x_mp) + fn(x_mm)) / (4.0 * dx * dx)
            hess[i, j] = d2
            hess[j, i] = d2

    return hess


def newton_raphson(
    fn: Callable[[float], float],
    x0: float,
    tol: float = 1e-8,
    max_iter: int = 100,
    dfn: Callable[[float], float] | None = None,
) -> float:
    """Find a root of *fn* using the Newton-Raphson method.

    Parameters
    ----------
    fn : callable
        Function ``f(x) -> float`` whose root is sought.
    x0 : float
        Initial guess.
    tol : float, optional
        Convergence tolerance on ``|f(x)|`` (default 1e-8).
    max_iter : int, optional
        Maximum number of iterations (default 100).
    dfn : callable or None, optional
        Derivative ``f'(x)``.  If ``None``, a central-difference
        approximation is used.

    Returns
    -------
    float
        Approximate root.

    Raises
    ------
    RuntimeError
        If the method does not converge within *max_iter* iterations.
    """
    x = float(x0)
    dx_fd = 1e-8

    for _ in range(max_iter):
        fx = fn(x)
        if abs(fx) < tol:
            return x

        if dfn is not None:
            fpx = dfn(x)
        else:
            fpx = (fn(x + dx_fd) - fn(x - dx_fd)) / (2.0 * dx_fd)

        if fpx == 0.0:
            raise RuntimeError("Zero derivative encountered in Newton-Raphson.")

        x = x - fx / fpx

    raise RuntimeError(
        f"Newton-Raphson did not converge after {max_iter} iterations "
        f"(|f(x)| = {abs(fn(x)):.2e})."
    )


def bisection(
    fn: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> float:
    """Find a root of *fn* on ``[a, b]`` using the bisection method.

    Parameters
    ----------
    fn : callable
        Continuous function ``f(x) -> float``.
    a : float
        Left endpoint of the bracket.
    b : float
        Right endpoint of the bracket.
    tol : float, optional
        Convergence tolerance on the bracket width (default 1e-8).
    max_iter : int, optional
        Maximum number of iterations (default 100).

    Returns
    -------
    float
        Approximate root.

    Raises
    ------
    ValueError
        If ``f(a)`` and ``f(b)`` have the same sign.
    RuntimeError
        If the method does not converge within *max_iter* iterations.
    """
    fa, fb = fn(a), fn(b)
    if fa * fb > 0:
        raise ValueError(
            f"f(a) and f(b) must have opposite signs; got f({a})={fa}, f({b})={fb}."
        )

    for _ in range(max_iter):
        mid = (a + b) / 2.0
        fm = fn(mid)
        if abs(fm) < tol or (b - a) / 2.0 < tol:
            return mid
        if fa * fm < 0:
            b = mid
            fb = fm
        else:
            a = mid
            fa = fm

    raise RuntimeError(f"Bisection did not converge after {max_iter} iterations.")


def trapezoidal_integration(
    fn: Callable[[float], float],
    a: float,
    b: float,
    n: int = 1000,
) -> float:
    """Numerical integration using the trapezoidal rule.

    Parameters
    ----------
    fn : callable
        Function to integrate.
    a : float
        Lower bound.
    b : float
        Upper bound.
    n : int, optional
        Number of trapezoids (default 1000).

    Returns
    -------
    float
        Approximate value of the definite integral.
    """
    x = np.linspace(a, b, n + 1)
    y = np.array([fn(xi) for xi in x])
    h = (b - a) / n
    return float(h * (y[0] / 2.0 + y[-1] / 2.0 + np.sum(y[1:-1])))


def monte_carlo_integration(
    fn: Callable[..., float],
    bounds: list[tuple[float, float]],
    n_samples: int = 100_000,
    seed: int | None = None,
) -> dict[str, float]:
    """Monte Carlo integration over a hyper-rectangular domain.

    Parameters
    ----------
    fn : callable
        Function to integrate.  Should accept a 1-D array of length
        ``len(bounds)`` (or individual floats for 1-D).
    bounds : list of (float, float)
        Integration bounds for each dimension, e.g. ``[(0, 1), (0, 2)]``.
    n_samples : int, optional
        Number of random samples (default 100 000).
    seed : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        ``estimate``  – estimated value of the integral.
        ``std_error`` – standard error of the estimate.
    """
    rng = np.random.default_rng(seed)
    d = len(bounds)
    bounds_arr = np.asarray(bounds, dtype=float)
    lows = bounds_arr[:, 0]
    highs = bounds_arr[:, 1]
    volume = float(np.prod(highs - lows))

    # Sample uniformly in the domain
    samples = rng.uniform(lows, highs, size=(n_samples, d))

    values = np.array([fn(s) for s in samples])
    estimate = volume * np.mean(values)
    std_error = volume * np.std(values, ddof=1) / np.sqrt(n_samples)

    return {
        "estimate": float(estimate),
        "std_error": float(std_error),
    }
