"""Supervised warm-start of the coordinator head (IMPROVEMENTS.md #2).

The sep-CMA-ES search starts from a zero head (``params.initial_theta`` -> uniform
policy), so the optimizer must rediscover routing from a single noisy binary bit
averaged over a minibatch. This module gives it a much better starting point: it
fits the head's **agent-selection rows** by supervised softmax cross-entropy /
InfoNCE against per-(query, model) correctness labels we already collected (the
oracle-ceiling matrices), then packs that into a CMA-ES initial mean ``x0`` via
:func:`trinity.coordinator.params.pack`.

Design contract (mirrors oracle_ceiling.py): the fit + pack are **pure numpy**
with NO torch dependency, so they unit-test on a box without a GPU. The only
torch/GPU step is encoding the queries with the frozen SLM, which lives behind a
lazily-imported helper (:func:`encode_queries`) that the tests never import.

Head layout (see coordinator/head.py / params.py): ``W`` is ``(n_a, d_h)`` with
rows ``[:n_models]`` = agent logits (which LLM) and ``[n_models:]`` = role logits.
We fit only the agent rows; role rows stay 0 (uniform role policy) and SVF scales
stay 1.0 (identity), so CMA-ES still learns role assignment + SVF from there.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import params as P

__all__ = [
    "load_labels",
    "fit_agent_head",
    "pack_warmstart_theta",
    "load_warmstart_theta",
    "encode_queries",
]


def load_warmstart_theta(path: str, spec: "P.ParamSpec") -> np.ndarray:
    """Load a warm-start theta .npy and validate its length against ``spec``.

    Returns a float64 vector of length ``spec.n_total``. Raises ``ValueError`` on a
    length mismatch (a layout error; a silent reshape would corrupt the head/SVF
    split). Pure numpy so train.py's warm-start path is unit-testable without torch.
    """
    theta = np.load(path).astype(np.float64).ravel()
    if theta.size != spec.n_total:
        raise ValueError(
            f"warm-start theta at {path} has {theta.size} params, expected "
            f"spec.n_total={spec.n_total}; layout mismatch, refusing to use it"
        )
    return theta


# ---------------------------------------------------------------------------
# Labels (from the oracle_ceiling matrix schema; no new API calls)
# ---------------------------------------------------------------------------
def load_labels(matrix_path: str) -> tuple[list[str], np.ndarray, list[str]]:
    """Read an ``oracle_matrix_<bench>.json`` into (query_ids, solve_prob, models).

    ``solve_prob[i, m]`` is model ``m``'s solve rate on query ``i`` = mean over the
    K samples. Models are ordered as they appear in the first task's ``per_model``.
    """
    matrix = json.loads(Path(matrix_path).read_text())
    tasks = matrix["tasks"]
    if not tasks:
        return [], np.zeros((0, 0)), []
    models = list(tasks[0]["per_model"].keys())
    qids: list[str] = []
    rows: list[list[float]] = []
    for t in tasks:
        qids.append(str(t.get("id", t.get("task_id", len(qids)))))
        rows.append([float(np.mean(t["per_model"][m])) for m in models])
    return qids, np.asarray(rows, dtype=float), models


# ---------------------------------------------------------------------------
# Pure-numpy fit of the agent-selection head rows
# ---------------------------------------------------------------------------
def _softmax(Z: np.ndarray) -> np.ndarray:
    Z = Z - Z.max(axis=1, keepdims=True)
    e = np.exp(Z)
    return e / e.sum(axis=1, keepdims=True)


def fit_agent_head(
    encodings: np.ndarray,
    solve_prob: np.ndarray,
    *,
    n_models: int,
    steps: int = 400,
    lr: float = 0.5,
    l2: float = 1e-3,
    tau: float = 1.0,
    prefer_disagree: bool = True,
    target_temp: float = 0.5,
    seed: int = 0,
    return_history: bool = False,
):
    """Fit agent-selection weights ``W_agent`` ``(n_models, d_h)`` by weighted CE.

    For each query, the supervised target is a categorical over models derived from
    the solve rates (peaked by ``target_temp`` so mass concentrates on the model(s)
    most likely to solve it). The loss is the cross-entropy between ``softmax(W·h)``
    and that target, optionally weighting each query by how much the models DISAGREE
    on it (``prefer_disagree``) since unanimous queries carry no routing signal. Full
    batch gradient descent (the gradient of softmax-CE is ``(softmax(z) - target)``),
    with L2 regularization to keep ``W`` at a modest scale that CMA-ES can refine.

    Args:
        encodings: ``(N, d_h)`` query features (L2-normalized SLM hidden states).
        solve_prob: ``(N, n_models)`` per-(query, model) solve rate in ``[0, 1]``.
        n_models: number of pool models (agent rows to fit).
        steps, lr, l2, tau: optimization hyperparameters.
        prefer_disagree: weight queries by cross-model solve-rate variance.
        target_temp: softmax temperature applied to solve rates to build the target
            (smaller -> peakier, routes harder toward the single best model).
        return_history: also return the per-step loss list.

    Returns:
        ``W_agent`` ``(n_models, d_h)`` (float64); or ``(W_agent, losses)`` if
        ``return_history``.
    """
    H = np.ascontiguousarray(encodings, dtype=float)
    sp = np.clip(np.asarray(solve_prob, dtype=float), 0.0, 1.0)
    if H.ndim != 2:
        raise ValueError(f"encodings must be 2D (N, d_h); got {H.shape}")
    N, d_h = H.shape
    if sp.shape != (N, n_models):
        raise ValueError(f"solve_prob must be (N={N}, n_models={n_models}); got {sp.shape}")

    # Supervised target: a peaked categorical over models from the solve rates.
    # Queries no model solves get a uniform (signal-free) target; they are also
    # down-weighted below because their cross-model variance is ~0.
    row_sum = sp.sum(axis=1, keepdims=True)
    has_solver = (row_sum.squeeze(-1) > 0)
    target = np.full((N, n_models), 1.0 / n_models, dtype=float)
    if target_temp and target_temp > 0:
        peaked = _softmax(sp / target_temp)
    else:
        peaked = sp / np.maximum(row_sum, 1e-9)
    target[has_solver] = peaked[has_solver]

    # Per-query weights: emphasize queries where models disagree (real routing signal).
    if prefer_disagree:
        var = sp.var(axis=1)
        w = 0.1 + var / (var.mean() + 1e-12)  # floor so every query contributes a little
    else:
        w = np.ones(N)
    w = w * (N / w.sum())          # normalize to mean 1
    wn = w / N                     # per-query gradient weight (sums to 1)

    rng = np.random.default_rng(seed)
    W = rng.normal(0.0, 1e-3, size=(n_models, d_h))  # tiny non-zero break-symmetry init
    losses: list[float] = []
    for _ in range(steps):
        Z = (H @ W.T) / tau                       # (N, n_models)
        Pm = _softmax(Z)                          # (N, n_models)
        ce = -(target * np.log(Pm + 1e-12)).sum(axis=1)   # (N,)
        losses.append(float((ce * w).sum() / N + 0.5 * l2 * float((W * W).sum())))
        grad = ((Pm - target) * wn[:, None]).T @ H / tau + l2 * W   # (n_models, d_h)
        W -= lr * grad
    return (W, losses) if return_history else W


# ---------------------------------------------------------------------------
# Pack into a CMA-ES initial mean theta
# ---------------------------------------------------------------------------
def pack_warmstart_theta(W_agent: np.ndarray, spec: P.ParamSpec) -> np.ndarray:
    """Place ``W_agent`` into the head's agent rows; role rows = 0, SVF = 1.

    Returns a full-length ``theta`` (``spec.n_total``) suitable as the sep-CMA-ES
    initial mean. Role logits stay uniform and SVF stays identity so CMA-ES still
    learns those from the warm start.
    """
    n_a, d_h = spec.head_shape
    W_agent = np.asarray(W_agent, dtype=float)
    n_models = W_agent.shape[0]
    if W_agent.shape != (n_models, d_h) or n_models > n_a:
        raise ValueError(f"W_agent {W_agent.shape} incompatible with head {spec.head_shape}")
    W_full = np.zeros((n_a, d_h), dtype=float)
    W_full[:n_models] = W_agent
    svf = np.ones(spec.n_svf, dtype=float)
    return P.pack(W_full, svf)


# ---------------------------------------------------------------------------
# GPU encoding (torch; lazily imported, NOT touched by the numpy unit tests)
# ---------------------------------------------------------------------------
def encode_queries(prompts, *, model_name, device="cuda:0", dtype="bfloat16",
                   target_layer=26, l2_normalize=True, instruction=None):
    """Encode a list of query strings into ``(N, d_h)`` features with the frozen SLM.

    Imports torch + the CoordinatorEncoder lazily so this module stays import-clean
    on a torch-free box. Runs on the GPU for the actual warm-start; the returned
    array is what :func:`fit_agent_head` consumes (cache it to .npy on the box).

    ``target_layer`` is accepted for caller compatibility but NOT passed to the
    encoder: CoordinatorEncoder reads a fixed penultimate-token hidden state (the
    layer-26 ``target_layer`` is the SVF adapter's concern, not the feature's).
    """
    from .slm import CoordinatorEncoder  # lazy: pulls torch only when actually encoding

    enc = CoordinatorEncoder(model_name=model_name, device=device, dtype=dtype,
                             l2_normalize=l2_normalize)
    feats = []
    for q in prompts:
        text = (instruction + q) if instruction else q
        feats.append(np.asarray(enc.encode(text), dtype=float).reshape(-1))
    return np.vstack(feats)
