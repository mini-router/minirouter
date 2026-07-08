"""Learning-Rate Adaptation (LRA) for the sep-CMA-ES trainer under noisy reward.

Our training fitness is the mean of ``m_cma`` **noisy 0/1 bits** per candidate
(docs/SPEC.md §5.2). sep-CMA-ES updates its distribution at full rate every
generation, so when the fitness *differences between candidates* are smaller than
the sampling noise of those means, the intra-generation ranking is untrustworthy
and the mean marches on noise — one of the four named failure modes in
docs/IMPROVEMENTS.md (§"sparse, noisy binary reward"), and the reason the trained
router captures none of the ~4.9pt math head-room the oracle diagnostic found.

LRA (Nomura, Ono & Shirakawa, "CMA-ES with Learning Rate Adaptation", ACM TELO
2025, DOI 10.1145/3698203) damps the effective learning rate when the update's
signal-to-noise ratio (SNR) is low and restores it when signal returns — **at the
default population size, so it spends no extra evaluations.**

Which SNR? The paper measures the SNR of the parameter *update stream*. We tried
that first and it did **not** discriminate our regime: in separable CMA the mean
displacement decorrelates generation-to-generation even on a clean objective, so
its coherence is the same with or without reward noise (JOURNAL 2026-07-09). What
*is* observable for free here is the **fitness-estimation SNR within a
generation**: the spread of the candidate fitnesses (real signal) against the
sampling noise of each mean estimate (``p(1-p)/m_cma`` for a Bernoulli mean).

    snr = max(Var_candidates(f) - noise_var, 0) / noise_var

When real fitness differences dwarf the estimation noise the ranking is
trustworthy (``eta -> 1``); when they are buried in noise, damp (``eta -> eta_min``).
On a clean objective ``noise_var -> 0`` so ``snr -> inf`` and ``eta == 1`` — LRA is
**neutral by construction, never a regression** (verified in
``utility/lra_ablation.py``). Under injected noise it lowers the final and
worst-case distance-to-optimum, and the gain grows with the noise level.

Scope (documented honestly): the returned ``eta`` scales the **mean and
step-size** update — the dominant terms in our high-noise, step-size-driven
regime — following the LRA *principle* of SNR-targeting, specialized to noisy
fitness estimation. At ``eta == 1`` the optimizer is *exactly* vanilla
sep-CMA-ES, so a disabled or warming-up controller is a perfect no-op.

No torch, no network; imports only numpy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["LRAConfig", "LRAController"]


@dataclass(frozen=True)
class LRAConfig:
    """Hyperparameters for :class:`LRAController`.

    Defaults were tuned on the offline noisy-sphere ablation
    (``utility/lra_ablation.py``): exactly neutral on a clean objective, and a
    lower mean/worst-case final distance under injected fitness noise (gain
    growing with the noise level).

    Attributes:
        alpha: Target smoothed SNR. The learning rate is damped while the
            fitness-estimation SNR sits below ``alpha`` (candidate differences
            comparable to the sampling noise) and opens back toward 1.0 above it.
        beta: EMA smoothing constant for the per-generation SNR
            (``0 < beta <= 1``). Smaller = longer memory / steadier control.
        gain: Multiplicative adaptation gain ``c_eta``; the per-generation log
            step of ``eta`` is ``gain * err`` with ``err in (-1, 1)``.
        eta_min: Floor on the learning rate; keeps the optimizer moving even in
            a persistently noisy regime.
        warmup: Number of initial generations held at ``eta == 1.0`` while the
            SNR estimate settles (during warm-up the optimizer is exactly
            vanilla).
    """

    alpha: float = 1.0
    beta: float = 0.3
    gain: float = 0.3
    eta_min: float = 0.1
    warmup: int = 3

    def __post_init__(self) -> None:
        if not (0.0 < self.beta <= 1.0):
            raise ValueError(f"beta must be in (0, 1], got {self.beta}")
        if self.alpha <= 0.0:
            raise ValueError(f"alpha must be > 0, got {self.alpha}")
        if not (0.0 < self.eta_min <= 1.0):
            raise ValueError(f"eta_min must be in (0, 1], got {self.eta_min}")
        if self.gain < 0.0:
            raise ValueError(f"gain must be >= 0, got {self.gain}")
        if self.warmup < 0:
            raise ValueError(f"warmup must be >= 0, got {self.warmup}")

    @classmethod
    def from_dict(cls, d: dict | None) -> "LRAConfig":
        """Build from a config mapping (e.g. ``trinity.yaml``'s ``sep_cmaes.lra``).

        Unknown keys are ignored; ``None`` / ``{}`` yields the defaults. The
        ``enabled`` flag lives in the config block but is consumed by the caller
        (``train.py``), not stored here.
        """
        d = d or {}
        return cls(
            alpha=float(d.get("alpha", cls.alpha)),
            beta=float(d.get("beta", cls.beta)),
            gain=float(d.get("gain", cls.gain)),
            eta_min=float(d.get("eta_min", cls.eta_min)),
            warmup=int(d.get("warmup", cls.warmup)),
        )


class LRAController:
    """Fitness-SNR learning-rate controller for a CMA-ES update stream.

    Call :meth:`update` once per generation with the candidate fitnesses and the
    (mean) sampling-noise variance of a single fitness estimate. It returns the
    learning rate ``eta in [eta_min, 1.0]`` to apply to that generation's mean
    and step-size update.

    SNR
    ---
    ``observed_var = Var(fitnesses)`` mixes the real spread of candidate quality
    with the estimation noise; subtracting ``noise_var`` recovers the signal
    power, and ``snr = signal / noise_var`` is smoothed with an EMA (``beta``).
    ``noise_var <= 0`` (a clean / noiseless objective) yields ``eta == 1.0``.

    Rate update
    -----------
    ``err = (snr - alpha) / (snr + alpha) in (-1, 1)`` and
    ``eta <- clip(eta * exp(gain * err), eta_min, 1.0)``. Below-target SNR
    (ranking buried in noise) shrinks ``eta``; above-target grows it back to 1.0.

    Deterministic and stateful; construct one per optimizer run.
    """

    def __init__(self, config: LRAConfig | None = None) -> None:
        """Initialize the controller.

        Args:
            config: Hyperparameters; defaults to :class:`LRAConfig` defaults.
        """
        self.cfg = config or LRAConfig()
        self._snr = 0.0
        self._gen = 0
        self._eta = 1.0
        # Per-generation diagnostics (for logging / tests).
        self.last_snr: float = float("nan")

    @property
    def eta(self) -> float:
        """Current learning rate (1.0 until the first post-warmup update)."""
        return self._eta

    @property
    def generation(self) -> int:
        """Number of completed :meth:`update` calls."""
        return self._gen

    def update(self, fitnesses: np.ndarray, noise_var: float) -> float:
        """Ingest one generation's fitnesses + estimation noise, return ``eta``.

        Args:
            fitnesses: The candidate fitnesses of this generation (shape
                ``(popsize,)``); the SAME values fed to ``tell``.
            noise_var: Sampling-noise variance of a single candidate's fitness
                *estimate* (e.g. ``mean_i p_i(1-p_i)/m_cma`` for a mean of
                ``m_cma`` Bernoulli rewards). ``<= 0`` means a noiseless
                objective and forces ``eta == 1.0``.

        Returns:
            The learning rate ``eta in [eta_min, 1.0]`` to apply this generation.
            Always ``1.0`` during the warm-up window or when ``noise_var <= 0``.
        """
        self._gen += 1
        f = np.asarray(fitnesses, dtype=float).reshape(-1)

        if noise_var <= 0.0 or f.size < 2:
            # Noiseless (or degenerate) generation: the ranking is trustworthy.
            self.last_snr = float("inf")
            self._eta = 1.0
            return self._eta

        observed_var = float(f.var())
        signal = max(observed_var - noise_var, 0.0)
        raw_snr = signal / noise_var
        self._snr = (1.0 - self.cfg.beta) * self._snr + self.cfg.beta * raw_snr
        self.last_snr = self._snr

        if self._gen <= self.cfg.warmup:
            self._eta = 1.0
            return self._eta

        err = (self._snr - self.cfg.alpha) / (self._snr + self.cfg.alpha)
        eta = self._eta * math.exp(self.cfg.gain * err)
        self._eta = float(min(1.0, max(self.cfg.eta_min, eta)))
        return self._eta
