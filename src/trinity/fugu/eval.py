"""Honest evaluation of a Conductor: PURE binary correctness, with cost.

Reports only :func:`trinity.fugu.reward.is_correct` (never the shaped training
reward), so a number here cannot be inflated by partial credit. Supports several
reps per task: the single-sample noise that, per docs/RESULTS.md, swung random
routing by about 6 points is denoised by averaging reps, and the per-query
binary it emits feeds straight into ``scripts/oracle_ceiling.py`` (which supplies
the winner's-curse-debiased routing ceiling and bootstrap CIs, the FP/FN-proof
verdict layer).
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from trinity.fugu.cost import CostMeter, price_table
from trinity.fugu.reward import is_correct
from trinity.fugu.workflow import propose_and_run
from trinity.types import Task

__all__ = ["EvalResult", "evaluate"]


@dataclass
class EvalResult:
    """Aggregate + per-task evaluation outcome with cost."""

    n_tasks: int
    reps: int
    accuracy: float                       # mean over tasks of (mean over reps)
    parse_rate: float                     # mean fraction of proposals that parsed
    per_task: dict = field(default_factory=dict)
    per_query_binary: dict = field(default_factory=dict)   # task_id -> 0/1 majority
    cost: dict = field(default_factory=dict)
    aborted: bool = False


async def evaluate(
    conductor,
    tasks: list[Task],
    pool,
    pool_models: list[str],
    *,
    reps: int = 1,
    max_depth: int = 1,
    temperature: float = 0.2,
    prices: dict | None = None,
    cap_usd: float = 0.0,
    concurrency: int = 8,
    client=None,
) -> EvalResult:
    """Evaluate ``conductor`` on ``tasks`` with ``reps`` samples each.

    Uses sampling when ``reps > 1`` (so the reps are independent draws), greedy
    otherwise. Tasks run with bounded ``concurrency`` (the Fireworks pool's own
    semaphore still caps in-flight calls). Respects a spend ``cap_usd`` (0
    disables it): once the running spend crosses the cap, no NEW task is started
    and the result is returned with ``aborted=True`` and only the tasks finished
    so far, rather than overspending. The meter is mutated only between awaits in
    a single event loop, so the accounting stays consistent without locks.
    """
    meter = CostMeter(prices=prices or price_table(), cap_usd=cap_usd)
    per_task: dict[str, dict] = {}
    per_query_binary: dict[str, int] = {}
    aborted = {"v": False}
    sem = asyncio.Semaphore(max(1, concurrency))

    async def _one(task: Task) -> None:
        if aborted["v"]:
            return
        async with sem:
            votes: list[int] = []
            parsed: list[int] = []
            for _ in range(reps):
                if aborted["v"]:
                    break
                run = await propose_and_run(
                    conductor, task, pool, pool_models,
                    sample=(reps > 1), max_depth=max_depth,
                    temperature=temperature, reasoning="minimal", client=client,
                )
                meter.add_run(run)
                votes.append(is_correct(run, task))
                parsed.append(int(run.parsed_ok))
                if meter.aborted:
                    aborted["v"] = True
                    break
            if votes:
                per_task[task.task_id] = {
                    "acc": sum(votes) / len(votes),
                    "reps_correct": votes,
                    "parse_rate": sum(parsed) / len(parsed),
                }
                # Strict majority: a tie (2*sum == len) resolves to 0, not 1.
                # `>=` counted a 50/50 ballot as solved, which is partial credit
                # this harness explicitly must not emit -- and it is reachable via
                # even --reps or an odd --reps ballot truncated by the spend cap.
                # Resolving ties against the router is the conservative choice.
                per_query_binary[task.task_id] = int(2 * sum(votes) > len(votes))

    await asyncio.gather(*[_one(t) for t in tasks])

    task_accs = [v["acc"] for v in per_task.values()]
    parse_rates = [v["parse_rate"] for v in per_task.values()]
    accuracy = float(sum(task_accs) / len(task_accs)) if task_accs else 0.0
    parse_rate = float(sum(parse_rates) / len(parse_rates)) if parse_rates else 0.0
    return EvalResult(
        n_tasks=len(per_task),
        reps=reps,
        accuracy=accuracy,
        parse_rate=parse_rate,
        per_task=per_task,
        per_query_binary=per_query_binary,
        cost=meter.report(),
        aborted=aborted["v"],
    )
