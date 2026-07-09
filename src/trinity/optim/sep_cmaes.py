"""Separable (diagonal-covariance) CMA-ES wrapper for TRINITY training.

Thin, deterministic wrapper around the `cma` library run in SEPARABLE mode
(``CMA_diagonal=True``). The optimizer searches the joint parameter vector
``theta`` of dimension ``n = 13,312`` (= 6,144 linear-head params + 7,168 SVF
singular-value scales; see docs/SPEC.md §0.2). The objective is the mean binary
reward ``J(theta) = E[R(tau)]`` over a minibatch of task instances and is
**maximized**; the `cma` library minimizes, so fitnesses are negated internally.

Design notes (docs/SPEC.md §5):

* Population ``lambda`` defaults to ``ceil(4 + 3 ln n)`` (n=13312 -> 33).
* Parents ``mu = floor(lambda/2)``, default log recombination weights, and all
  other strategy constants come from the library's separable defaults.
* Initial mean ``x0`` is supplied by the caller (head W=0, SVF scales=1.0); a
  zeros vector is used if ``x0`` is None.
* ``sigma0 = 0.1`` by default; the coordinator L2-normalizes the hidden state so
  this step size stays well-behaved at the W=0 start.

This module imports no torch and runs on CPU only. The expensive fitness
function (real SLM + real pool LLMs) is injected by the caller via :func:`run`
or driven manually through :meth:`ask` / :meth:`tell`.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

# pycma treats seed==0 as "unseeded" (it ignores it), so a seed-0 run is not
# reproducible. We remap 0 to this fixed nonzero value in pycma's valid range
# [1, 2**32-1] so the documented reproducible default actually holds.
_SEED0_REMAP: int = 0x5EED  # arbitrary fixed constant; any nonzero value works.


def _import_cma():
    """Import pycma lazily, so this module (and trinity.optim) imports cleanly on
    boxes without `cma` — it is only needed when an optimizer is actually built."""
    try:
        import cma  # type: ignore

        return cma
    except ImportError as exc:  # pragma: no cover - exercised only when missing.
        raise ImportError(
            "The 'cma' package (pycma) is required to build SepCMAES. "
            "Install it with:  pip install cma"
        ) from exc


def default_popsize(n: int) -> int:
    """Return the CMA-ES default population size ``lambda``.

    ``lambda = ceil(4 + 3 * ln(n))``. For the TRINITY joint dimension
    ``n = 13,312`` this evaluates to ``ceil(32.49...) = 33`` (docs/SPEC.md §0.2).

    Args:
        n: Search-space dimension (number of free parameters).

    Returns:
        The population size as a positive integer.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    return int(math.ceil(4 + 3 * math.log(n)))


class SepCMAES:
    """Separable CMA-ES optimizer that **maximizes** a scalar objective.

    Wraps :class:`cma.CMAEvolutionStrategy` with ``CMA_diagonal=True`` so the
    covariance matrix stays diagonal (the separable variant of Ros & Hansen,
    2008). All public fitness values are interpreted as quantities to maximize;
    the negation required by the minimizing backend is handled internally.

    Example (manual ask/tell loop)::

        opt = SepCMAES(n=13312, sigma0=0.1, seed=0, maxiter=60)
        while not opt.stop():
            solutions = opt.ask()
            fitnesses = [objective(x) for x in solutions]  # higher = better
            opt.tell(solutions, fitnesses)
        best_x, best_f = opt.best()
    """

    def __init__(
        self,
        n: int,
        sigma0: float = 0.1,
        x0: np.ndarray | None = None,
        popsize: int | None = None,
        seed: int = 0,
        maxiter: int = 60,
    ) -> None:
        """Initialize the separable CMA-ES strategy.

        Args:
            n: Search-space dimension (TRINITY: 13,312).
            sigma0: Initial step size (TRINITY default 0.1).
            x0: Initial mean vector of shape ``(n,)``. Defaults to ``zeros(n)``
                (head W=0, SVF scales should be added by the caller's packing if
                a non-zero identity start is desired).
            popsize: Population size ``lambda``. If None, computed via
                :func:`default_popsize` (n=13312 -> 33).
            seed: RNG seed for reproducible sampling.
            maxiter: Maximum number of generations ``T`` (TRINITY default 60).

        Raises:
            ValueError: If ``x0`` is provided with a shape other than ``(n,)``.
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        self.n: int = int(n)
        self.sigma0: float = float(sigma0)
        self.seed: int = int(seed)
        self.maxiter: int = int(maxiter)
        self._popsize: int = int(popsize) if popsize is not None else default_popsize(n)

        if x0 is None:
            x0_vec = np.zeros(self.n, dtype=float)
        else:
            x0_vec = np.asarray(x0, dtype=float).reshape(-1)
            if x0_vec.shape != (self.n,):
                raise ValueError(
                    f"x0 must have shape ({self.n},), got {x0_vec.shape}"
                )

        # pycma IGNORES seed==0 ("cma ignores if seed==0") and leaves numpy's
        # global RNG unseeded, so the documented reproducible default would in
        # fact be non-reproducible. Remap 0 to a fixed nonzero value in pycma's
        # valid range [1, 2**32-1], leaving every other seed unchanged, so the
        # contract holds. self.seed keeps the caller's original value.
        cma_seed = self.seed if self.seed != 0 else _SEED0_REMAP

        # `verbose=-9` silences pycma's stdout/file logging. Strategy constants
        # (c_sigma, d_sigma, c_1, c_mu, mu, recombination weights) all use the
        # library's separable defaults per docs/SPEC.md §5.3.
        opts = {
            "CMA_diagonal": True,
            "popsize": self._popsize,
            "seed": cma_seed,
            "maxiter": self.maxiter,
            "verbose": -9,
        }
        cma = _import_cma()
        self._es = cma.CMAEvolutionStrategy(list(x0_vec), self.sigma0, opts)

        # Track the best-so-far in MAXIMIZATION space (so callers never see the
        # internal sign flip). None until the first `tell`.
        self._best_x: np.ndarray | None = None
        self._best_f: float = -math.inf

    # ------------------------------------------------------------------ #
    # Core ask / tell interface
    # ------------------------------------------------------------------ #
    def ask(self) -> list[np.ndarray]:
        """Sample a new population of candidate solutions.

        Returns:
            A list of ``popsize`` numpy arrays, each of shape ``(n,)``.
        """
        return [np.asarray(x, dtype=float) for x in self._es.ask()]

    def tell(
        self,
        solutions: list[np.ndarray],
        fitnesses: list[float],
    ) -> None:
        """Update the distribution from evaluated candidates.

        Fitnesses are to be **maximized**. They are negated before being passed
        to the minimizing `cma` backend. The internal best-so-far is updated in
        maximization space.

        Args:
            solutions: The candidate vectors returned by :meth:`ask`.
            fitnesses: One scalar per solution; larger means better.

        Raises:
            ValueError: If the two lists differ in length.
        """
        if len(solutions) != len(fitnesses):
            raise ValueError(
                f"solutions ({len(solutions)}) and fitnesses "
                f"({len(fitnesses)}) must have equal length"
            )
        sols = [np.asarray(x, dtype=float) for x in solutions]
        fits = [float(f) for f in fitnesses]

        # cma minimizes -> feed negated objective.
        self._es.tell(sols, [-f for f in fits])

        # Update best-so-far in maximization space.
        for x, f in zip(sols, fits):
            if f > self._best_f:
                self._best_f = f
                self._best_x = x.copy()

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def best(self) -> tuple[np.ndarray, float]:
        """Return the best solution found so far and its (maximized) fitness.

        Prefers the library's own incumbent (``xbest``) when available, which is
        more robust than tracking the raw population best; falls back to the
        locally tracked best, then to the current distribution mean.

        Returns:
            A tuple ``(best_x, best_f)`` where ``best_x`` has shape ``(n,)`` and
            ``best_f`` is the objective value in maximization space.

        Raises:
            RuntimeError: If called before any :meth:`tell`.
        """
        lib_best = getattr(self._es.result, "xbest", None)
        lib_fval = getattr(self._es.result, "fbest", None)
        if lib_best is not None and lib_fval is not None:
            # `fbest` is in the minimized (negated) space -> flip back.
            return np.asarray(lib_best, dtype=float), -float(lib_fval)
        if self._best_x is not None:
            return self._best_x.copy(), self._best_f
        raise RuntimeError("best() called before any tell(); no evaluation yet.")

    def stop(self) -> bool:
        """Whether any CMA-ES termination criterion (e.g. ``maxiter``) is met.

        Returns:
            True if the optimizer should stop, else False.
        """
        return bool(self._es.stop())

    @property
    def popsize(self) -> int:
        """Population size ``lambda`` in use."""
        return self._popsize

    @property
    def iteration(self) -> int:
        """Number of completed generations (``tell`` calls)."""
        return int(self._es.countiter)


def run(
    objective: Callable[[np.ndarray], float],
    n: int,
    *,
    sigma0: float = 0.1,
    x0: np.ndarray | None = None,
    popsize: int | None = None,
    seed: int = 0,
    maxiter: int = 60,
    verbose: bool = False,
) -> tuple[np.ndarray, float, list[dict]]:
    """Run separable CMA-ES to **maximize** ``objective`` and log per-iteration.

    Standalone driver used by smoke test S7 (synthetic deterministic fitness)
    and by the training entrypoint. Each generation: ask -> evaluate every
    candidate with ``objective`` -> tell -> record the best.

    Args:
        objective: Callable mapping a parameter vector of shape ``(n,)`` to a
            scalar fitness to be MAXIMIZED.
        n: Search-space dimension.
        sigma0: Initial step size.
        x0: Initial mean vector of shape ``(n,)``; defaults to ``zeros(n)``.
        popsize: Population size ``lambda``; defaults to :func:`default_popsize`.
        seed: RNG seed.
        maxiter: Maximum number of generations ``T``.
        verbose: If True, print a one-line summary per generation.

    Returns:
        A tuple ``(best_x, best_f, history)`` where:

        * ``best_x``: best parameter vector found, shape ``(n,)``.
        * ``best_f``: its objective value (maximization space).
        * ``history``: list of per-iteration dicts with keys
          ``{"iteration", "best_fitness", "gen_best_fitness", "gen_mean_fitness"}``
          suitable for logging ``J(theta)`` over training (docs/SPEC.md §5.2).
    """
    opt = SepCMAES(
        n=n,
        sigma0=sigma0,
        x0=x0,
        popsize=popsize,
        seed=seed,
        maxiter=maxiter,
    )
    history: list[dict] = []

    while not opt.stop():
        solutions = opt.ask()
        fitnesses = [float(objective(x)) for x in solutions]
        opt.tell(solutions, fitnesses)

        _, best_f = opt.best()
        gen_best = max(fitnesses)
        gen_mean = float(np.mean(fitnesses))
        record = {
            "iteration": opt.iteration,
            "best_fitness": best_f,
            "gen_best_fitness": gen_best,
            "gen_mean_fitness": gen_mean,
        }
        history.append(record)
        if verbose:
            print(
                f"[sep-CMA-ES] iter {opt.iteration:3d}/{maxiter} | "
                f"best={best_f:+.4f} | gen_best={gen_best:+.4f} | "
                f"gen_mean={gen_mean:+.4f}"
            )

    best_x, best_f = opt.best()
    return best_x, best_f, history


if __name__ == "__main__":
    # Smoke test S7: optimize a synthetic deterministic objective at the real
    # TRINITY dimension and confirm J increases and lambda is configured.
    _N = 13312
    _rng = np.random.default_rng(0)
    _theta_star = _rng.standard_normal(_N) * 0.05

    def _sphere(x: np.ndarray) -> float:
        """Negative squared distance to a target (maximized at theta_star)."""
        d = x - _theta_star
        return -float(np.dot(d, d))

    _bx, _bf, _hist = run(_sphere, _N, maxiter=10, verbose=True)
    print(f"popsize (lambda) = {default_popsize(_N)}")
    print(f"final best fitness = {_bf:+.6f}")
    print(f"monotone increasing best_fitness: "
          f"{all(_hist[i]['best_fitness'] <= _hist[i + 1]['best_fitness'] + 1e-12 for i in range(len(_hist) - 1))}")
