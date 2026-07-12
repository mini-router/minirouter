"""The inner coordination loop: run one query through up to K turns.

This is provider/torch-agnostic glue. It takes:
  - a `policy` object exposing `decide(transcript_text, *, sample, rng) -> (agent_idx, Role)`
    (the real one is trinity.coordinator.policy.CoordinatorPolicy; tests pass a mock),
  - an async `pool` exposing `chat(model, messages, *, temperature, top_p, max_tokens)
    -> ChatResult` (trinity.llm.fireworks_client.FireworksPool; tests pass a stub),
so the whole loop can be exercised end-to-end with zero GPU and zero network (S4).

See docs/SPEC.md §2 (data-flow) and §4 (protocol). Termination rule:
  τ = min{ k ≤ K : R_k = Verifier ∧ verdict = ACCEPT (and a Worker output already exists) }
  else τ = K.  Final answer = O_τ.
"""
from __future__ import annotations

import asyncio
from typing import Protocol

from ..roles import postprocess as _pp
from ..roles import prompts as _prompts
from ..roles import verifier as _verifier
from ..types import Role, Task, Trajectory, TurnRecord


class Policy(Protocol):
    def decide(self, transcript_text: str, *, sample: bool, rng=None) -> tuple[int, Role]: ...


class TrajectoryError(RuntimeError):
    """Base class for run_trajectory failures with task/route context."""


class TrajectoryTimeoutError(TrajectoryError):
    """A single LLM request or an entire trajectory exceeded its timeout."""


def _describe_pool_model(pool, model_name: str) -> tuple[str | None, str]:
    describe = getattr(pool, "describe_model", None)
    if callable(describe):
        try:
            provider, resolved = describe(model_name)
            return str(provider), str(resolved)
        except Exception:
            pass
    return None, model_name


def _transcript_text(task: Task, turns: list[TurnRecord]) -> str:
    """Text fed to the coordinator SLM (query + all prior processed outputs).

    Kept self-contained so the SLM-input format does not couple to the roles
    module's prompt rendering.
    """
    parts = [f"QUERY:\n{task.prompt}"]
    for t in turns:
        parts.append(f"[Turn {t.turn} | {t.role.value} | {t.agent_name}]\n{t.processed_output}")
    return "\n\n".join(parts)


async def run_trajectory(
    task: Task,
    policy: Policy,
    pool,
    pool_models: list[str],
    *,
    max_turns: int = 5,
    sample: bool = False,
    rng=None,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    top_p: float = 1.0,
    reasoning: str | None = "minimal",
    request_timeout_s: float | None = None,
    verifier_requires_prior_worker: bool = True,
    client=None,
) -> Trajectory:
    """Run one trajectory τ. Returns a Trajectory (reward left None; score later)."""
    traj = Trajectory(task=task, turns=[])
    has_worker_output = False

    for k in range(1, max_turns + 1):
        ttext = _transcript_text(task, traj.turns)
        agent_idx, role = policy.decide(ttext, sample=sample, rng=rng)
        agent_name = pool_models[agent_idx % len(pool_models)]
        provider_name, resolved_model = _describe_pool_model(pool, agent_name)
        route_text = f"provider={provider_name or 'unknown'} model={resolved_model}"
        print(
            f"[traj] task={task.task_id} benchmark={task.benchmark} turn={k}/{max_turns} "
            f"role={role.value} agent={agent_name} {route_text} start",
            flush=True,
        )

        messages = _prompts.build_messages(role, task.prompt, traj.turns)
        kwargs = dict(temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        if client is not None:
            kwargs["client"] = client
        if reasoning is not None:
            kwargs["reasoning"] = reasoning
        try:
            chat_coro = pool.chat(agent_name, messages, **_filter_supported(pool.chat, kwargs))
            if request_timeout_s and request_timeout_s > 0:
                res = await asyncio.wait_for(chat_coro, timeout=float(request_timeout_s))
            else:
                res = await chat_coro
        except asyncio.TimeoutError as exc:
            message = (
                f"timeout waiting for {route_text} task={task.task_id} "
                f"benchmark={task.benchmark} turn={k}/{max_turns}"
            )
            print(f"[traj] !! {message}", flush=True)
            raise TrajectoryTimeoutError(message) from exc
        except Exception as exc:
            message = (
                f"error from {route_text} task={task.task_id} "
                f"benchmark={task.benchmark} turn={k}/{max_turns}: {type(exc).__name__}: {exc}"
            )
            print(f"[traj] !! {message}", flush=True)
            raise TrajectoryError(message) from exc

        raw = res.text
        processed = _pp.postprocess(raw, role)
        verdict = _verifier.parse_verdict(raw) if role == Role.VERIFIER else None

        traj.turns.append(
            TurnRecord(
                turn=k,
                agent_name=agent_name,
                role=role,
                raw_output=raw,
                processed_output=processed,
                verdict=verdict,
                prompt_tokens=getattr(res, "prompt_tokens", 0),
                completion_tokens=getattr(res, "completion_tokens", 0),
            )
        )
        print(
            f"[traj] task={task.task_id} benchmark={task.benchmark} turn={k}/{max_turns} "
            f"role={role.value} agent={agent_name} {route_text} done verdict={verdict or '-'}",
            flush=True,
        )
        if role == Role.WORKER:
            has_worker_output = True

        # Termination: Verifier ACCEPT, guarded by "a Worker output must already exist"
        # (SPEC §0.3.5 — prevents a turn-1 Verifier from accepting an empty solution).
        accept = verdict == "ACCEPT" and (has_worker_output or not verifier_requires_prior_worker)
        if accept:
            traj.terminated_by = "accept"
            break

    traj.final_answer = _final_answer(traj)
    return traj


def _final_answer(traj: Trajectory) -> str:
    """O_τ: prefer the last Worker output; fall back to the last non-verifier output."""
    for t in reversed(traj.turns):
        if t.role == Role.WORKER:
            return t.processed_output
    for t in reversed(traj.turns):
        if t.role != Role.VERIFIER:
            return t.processed_output
    return traj.turns[-1].processed_output if traj.turns else ""


def _filter_supported(fn, kwargs: dict) -> dict:
    """Drop kwargs the client doesn't accept (e.g. `reasoning` on a stub)."""
    import inspect

    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}
