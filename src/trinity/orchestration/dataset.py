"""Benchmark loaders + minibatch sampling for TRINITY training/eval.

This module turns raw benchmark datasets into the canonical
:class:`trinity.types.Task` objects consumed by the inner loop
(``orchestration/session.py``) and the reward checkers (``orchestration/reward.py``).

Design constraints (see docs/SPEC.md §6, §8):
- HuggingFace ``datasets`` is imported lazily and guarded. The LOCAL dev box has
  no network/GPU, so every loader has an OFFLINE fallback: a tiny hand-written
  toy set (2-3 Tasks per benchmark) so smoke tests (S4/S5) run with zero network.
- ``load_tasks`` is deterministic given ``seed``: shuffling/truncation use a seeded
  ``random.Random`` so two calls with the same arguments return identical lists.
- The ``answer`` field is whatever ``reward.score`` needs for that benchmark:
    * math500 / aime / gsm8k -> reference answer string (boxed-answer / last-number match)
    * mmlu / gpqa     -> the correct option LETTER ("A".."D")
    * livecodebench / humaneval -> a dict test spec {"tests": [...], "fn_name": ...}
    * bbh             -> raw gold target text (a "(X)" choice or free-form string)

Public API
----------
- ``load_tasks(benchmark, split, max_items, seed=0) -> list[Task]``
- ``sample_minibatch(tasks, m, rng) -> list[Task]``
- ``SUPPORTED_BENCHMARKS`` (tuple[str, ...])

The HuggingFace dataset ids used (when ``datasets`` + network are available):
- math500       : ``HuggingFaceH4/MATH-500`` (fallback ``qwedsacf/competition_math``)
- mmlu          : ``cais/mmlu`` (config ``all``)
- gpqa          : ``Idavidrein/gpqa`` (config ``gpqa_diamond``)
- livecodebench : ``lighteval/code_generation_lite`` (V1 train / V6 eval; parquet mirror)
- gsm8k         : ``openai/gsm8k`` (config ``main``)
- humaneval     : ``openai/openai_humaneval`` (single ``test`` split, 164 items)
- bbh           : ``lukaemon/bbh`` (27 subtask configs, see ``BBH_SUBTASKS``)
"""
from __future__ import annotations

import random
from typing import Any

from trinity.types import Task

__all__ = ["load_tasks", "sample_minibatch", "SUPPORTED_BENCHMARKS"]

SUPPORTED_BENCHMARKS: tuple[str, ...] = (
    "math500",
    "mmlu",
    "gpqa",
    "livecodebench",
    "gsm8k",
    "humaneval",
    "bbh",
)

# Letters used for multiple-choice option indexing (MMLU/GPQA).
_CHOICE_LETTERS: tuple[str, ...] = ("A", "B", "C", "D", "E", "F", "G", "H")

# The 27 canonical BIG-Bench Hard subtasks (each a separate ``lukaemon/bbh``
# dataset config). Fixed, well-known list -- see the BBH paper (Suzgun et al.,
# 2022) appendix / the ``lukaemon/bbh`` dataset card for the enumeration.
BBH_SUBTASKS: tuple[str, ...] = (
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "dyck_languages",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies",
    "word_sorting",
)


# --------------------------------------------------------------------------- #
# Lazy / guarded HuggingFace `datasets` import
# --------------------------------------------------------------------------- #
def _try_load_hf(
    path: str,
    *,
    name: str | None = None,
    split: str | None = None,
    version_tag: str | None = None,
) -> Any | None:
    """Attempt ``datasets.load_dataset``; return ``None`` on any failure.

    The import is lazy (so the module imports fine on a box without ``datasets``)
    and any error -- missing package, no network, unknown dataset id, gated repo --
    is swallowed so that callers fall back to the offline toy set. Failures are
    intentionally silent here; the caller decides whether the fallback is loud.

    Parameters
    ----------
    path:
        HuggingFace dataset repository id.
    name:
        Optional dataset config name (e.g. ``"all"`` for MMLU).
    split:
        Optional split string passed straight to ``load_dataset``.

    Returns
    -------
    The loaded dataset object, or ``None`` if loading was not possible.
    """
    try:
        from datasets import load_dataset  # type: ignore import-not-found
    except Exception:
        return None
    try:
        kwargs: dict[str, Any] = {}
        if name is not None:
            kwargs["name"] = name
        if split is not None:
            kwargs["split"] = split
        if version_tag is not None:
            kwargs["version_tag"] = version_tag
        return load_dataset(path, **kwargs)
    except Exception:
        return None


def _row_get(row: Any, *keys: str, default: Any = None) -> Any:
    """Return the first present key from a (dict-like) dataset row."""
    for k in keys:
        try:
            if k in row and row[k] is not None:
                return row[k]
        except TypeError:
            # Non-mapping row; give up.
            break
    return default


# --------------------------------------------------------------------------- #
# Per-benchmark HuggingFace parsers (return list[Task] or None on failure)
# --------------------------------------------------------------------------- #
def _load_math500_hf(split: str) -> list[Task] | None:
    """MATH-500 loader. answer = reference final answer string."""
    ds = _try_load_hf("HuggingFaceH4/MATH-500", split=split or "test")
    src = "HuggingFaceH4/MATH-500"
    if ds is None:
        # Fallback dataset uses a different schema (uses "solution" only).
        ds = _try_load_hf("qwedsacf/competition_math", split=split or "test")
        src = "qwedsacf/competition_math"
    if ds is None:
        return None
    tasks: list[Task] = []
    for i, row in enumerate(ds):
        problem = _row_get(row, "problem", "question", default="")
        answer = _row_get(row, "answer", "solution", default="")
        if not problem:
            continue
        tasks.append(
            Task(
                task_id=f"math500-{i}",
                benchmark="math500",
                prompt=str(problem),
                answer=str(answer),
                meta={
                    "source": src,
                    "subject": _row_get(row, "subject", "type"),
                    "level": _row_get(row, "level"),
                },
            )
        )
    return tasks or None


def _load_mmlu_hf(split: str) -> list[Task] | None:
    """MMLU loader. answer = correct option LETTER ("A".."D")."""
    ds = _try_load_hf("cais/mmlu", name="all", split=split or "test")
    if ds is None:
        return None
    tasks: list[Task] = []
    for i, row in enumerate(ds):
        question = _row_get(row, "question", default="")
        choices = _row_get(row, "choices", default=None)
        answer_idx = _row_get(row, "answer", default=None)
        if not question or not choices or answer_idx is None:
            continue
        try:
            answer_idx = int(answer_idx)
        except (TypeError, ValueError):
            continue
        if not (0 <= answer_idx < len(_CHOICE_LETTERS)):
            continue
        tasks.append(
            Task(
                task_id=f"mmlu-{i}",
                benchmark="mmlu",
                prompt=_format_mcq(str(question), list(choices)),
                answer=_CHOICE_LETTERS[answer_idx],
                meta={
                    "source": "cais/mmlu",
                    "subject": _row_get(row, "subject"),
                    "choices": list(choices),
                },
            )
        )
    return tasks or None


def _load_gpqa_hf(split: str) -> list[Task] | None:
    """GPQA-Diamond loader.

    GPQA stores the correct answer plus three distractors as separate columns.
    We shuffle them deterministically (per-row seeded by index) into A-D and
    record the resulting correct letter as the answer.
    """
    ds = _try_load_hf("Idavidrein/gpqa", name="gpqa_diamond", split=split or "train")
    if ds is None:
        return None
    tasks: list[Task] = []
    for i, row in enumerate(ds):
        question = _row_get(row, "Question", "question", default="")
        correct = _row_get(row, "Correct Answer", default=None)
        incorrect = [
            _row_get(row, "Incorrect Answer 1"),
            _row_get(row, "Incorrect Answer 2"),
            _row_get(row, "Incorrect Answer 3"),
        ]
        incorrect = [c for c in incorrect if c is not None]
        if not question or correct is None or len(incorrect) < 3:
            continue
        options = [str(correct)] + [str(c) for c in incorrect[:3]]
        # Deterministic per-row shuffle so option positions are stable.
        order = list(range(len(options)))
        random.Random(i).shuffle(order)
        shuffled = [options[j] for j in order]
        correct_pos = order.index(0)  # original index 0 == correct answer
        tasks.append(
            Task(
                task_id=f"gpqa-{i}",
                benchmark="gpqa",
                prompt=_format_mcq(str(question), shuffled),
                answer=_CHOICE_LETTERS[correct_pos],
                meta={
                    "source": "Idavidrein/gpqa",
                    "config": "gpqa_diamond",
                    "choices": shuffled,
                },
            )
        )
    return tasks or None


def _load_livecodebench_hf(split: str) -> list[Task] | None:
    """LiveCodeBench loader.

    Per SPEC §6.1 the in-distribution split is V1 (train, 400) and V6
    (eval, 175). We map ``split`` -> release version:
        "train" / "v1" -> release_v1
        "test"  / "v6" -> release_v6

    answer is a dict test spec consumed by the sandboxed pass@1 executor:
        {"tests": [{"input": str, "output": str}, ...],
         "fn_name": str | None,
         "starter_code": str | None}
    """
    version = _lcb_version_for_split(split)
    # The original livecodebench repo still ships a loading script, which modern
    # `datasets` versions reject. Prefer the parquet-backed mirror first.
    candidates: list[tuple[str, dict[str, str]]] = [
        ("lighteval/code_generation_lite", {"name": version, "split": "test"}),
        ("lighteval/code_generation_lite", {"split": version}),
        ("sam-paech/livecodebench-code_generation_lite", {"split": version}),
        ("livecodebench/code_generation_lite", {"name": version, "split": "test"}),
        ("livecodebench/code_generation_lite", {"split": version}),
    ]
    ds = None
    src = "lighteval/code_generation_lite"
    for path, kwargs in candidates:
        ds = _try_load_hf(path, **kwargs)
        if ds is not None:
            src = path
            break
    if ds is None:
        return None
    tasks: list[Task] = []
    for i, row in enumerate(ds):
        question = _row_get(
            row, "question_content", "question", "problem", default=""
        )
        if not question:
            continue
        tests = _parse_lcb_tests(row)
        tasks.append(
            Task(
                task_id=str(_row_get(row, "question_id", default=f"lcb-{i}")),
                benchmark="livecodebench",
                prompt=str(question),
                answer={
                    "tests": tests,
                    "fn_name": _row_get(row, "fn_name", "func_name"),
                    "starter_code": _row_get(row, "starter_code"),
                },
                meta={
                    "source": src,
                    "version": version,
                    "platform": _row_get(row, "platform"),
                    "difficulty": _row_get(row, "difficulty"),
                },
            )
        )
    return tasks or None


def _load_gsm8k_hf(split: str) -> list[Task] | None:
    """GSM8K loader. answer = the final numeric answer after the ``####`` marker."""
    ds = _try_load_hf("openai/gsm8k", name="main", split=split or "test")
    if ds is None:
        ds = _try_load_hf("gsm8k", name="main", split=split or "test")
    if ds is None:
        return None
    tasks: list[Task] = []
    for i, row in enumerate(ds):
        question = _row_get(row, "question", default="")
        raw_answer = _row_get(row, "answer", default="")
        if not question or not raw_answer:
            continue
        final = _gsm8k_final_answer(str(raw_answer))
        if final is None:
            continue
        tasks.append(
            Task(
                task_id=f"gsm8k-{i}",
                benchmark="gsm8k",
                prompt=f"{question}\n\nGive the final answer in \\boxed{{}}.",
                answer=final,
                meta={"source": "openai/gsm8k", "rationale": str(raw_answer)},
            )
        )
    return tasks or None


def _gsm8k_final_answer(answer_text: str) -> str | None:
    """Extract the ``#### <number>`` final answer GSM8K appends to every rationale."""
    marker = "####"
    idx = answer_text.rfind(marker)
    if idx == -1:
        return None
    tail = answer_text[idx + len(marker):].strip().replace(",", "")
    return tail or None


def _load_humaneval_hf(split: str) -> list[Task] | None:
    """HumanEval loader. answer = an assert-based test spec for ``run_pass_at_1``.

    HumanEval ships a single 164-item ``test`` split (no train split exists).
    Each row's ``test`` field defines a ``check(candidate)`` function; we wrap it
    into the same ``{"tests": [...]}`` contract the code-execution reward checker
    (``reward.run_pass_at_1``) already understands for assert-style tests, calling
    ``check(<entry_point>)`` against whatever the model defines in its completion.
    """
    ds = _try_load_hf("openai/openai_humaneval", split="test")
    if ds is None:
        ds = _try_load_hf("openai_humaneval", split="test")
    if ds is None:
        return None
    tasks: list[Task] = []
    for i, row in enumerate(ds):
        prompt = _row_get(row, "prompt", default="")
        entry_point = _row_get(row, "entry_point", default="")
        test_code = _row_get(row, "test", default="")
        task_id = _row_get(row, "task_id", default=f"HumanEval/{i}")
        if not prompt or not entry_point or not test_code:
            continue
        harness = f"{test_code}\ncheck({entry_point})\n"
        tasks.append(
            Task(
                task_id=str(task_id),
                benchmark="humaneval",
                prompt=_format_humaneval_prompt(str(prompt)),
                answer={
                    "tests": [harness],
                    "fn_name": entry_point,
                    "starter_code": str(prompt),
                },
                meta={"source": "openai/openai_humaneval", "entry_point": entry_point},
            )
        )
    return tasks or None


def _format_humaneval_prompt(signature: str) -> str:
    """Wrap a HumanEval function signature/docstring into a completion prompt."""
    return (
        f"{signature}\n\n"
        "Complete the function above. Reply with the FULL function definition "
        "(signature, docstring, and body) inside a single ```python code fence."
    )


def _load_bbh_hf(split: str) -> list[Task] | None:
    """BIG-Bench Hard loader across all 27 ``BBH_SUBTASKS`` configs.

    Each subtask is a separate ``lukaemon/bbh`` dataset config with a single
    ``test`` split. ``answer`` is the raw gold ``target`` text: either a
    ``"(X)"``-shaped multiple-choice letter (already embedded as lettered
    options inside ``input`` for those subtasks) or free-form text (a word,
    phrase, "True"/"False", etc.) for the rest. See ``reward._check_bbh``.
    """
    tasks: list[Task] = []
    for subtask in BBH_SUBTASKS:
        ds = _try_load_hf("lukaemon/bbh", name=subtask, split="test")
        if ds is None:
            continue
        for i, row in enumerate(ds):
            question = _row_get(row, "input", default="")
            target = _row_get(row, "target", default="")
            if not question or not target:
                continue
            tasks.append(
                Task(
                    task_id=f"bbh-{subtask}-{i}",
                    benchmark="bbh",
                    prompt=_format_bbh_prompt(str(question)),
                    answer=str(target).strip(),
                    meta={"source": "lukaemon/bbh", "subtask": subtask},
                )
            )
    return tasks or None


def _format_bbh_prompt(question: str) -> str:
    """Wrap a BBH question with an instruction for a single, parseable final line."""
    return (
        f"{question.strip()}\n\n"
        "End your response with a final line in the exact form:\n"
        "Answer: <answer>"
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _format_mcq(question: str, choices: list[Any]) -> str:
    """Render a multiple-choice question with lettered options.

    The prompt explicitly asks the pool model to end with a single answer
    letter so the reward checker's letter extraction is reliable.
    """
    lines = [question.strip(), ""]
    for letter, choice in zip(_CHOICE_LETTERS, choices):
        lines.append(f"{letter}. {choice}")
    lines.append("")
    lines.append("Answer with the single letter of the correct option.")
    return "\n".join(lines)


def _lcb_version_for_split(split: str) -> str:
    """Map a logical split string onto a LiveCodeBench release config name."""
    s = (split or "").strip().lower()
    if s in ("test", "eval", "v6", "release_v6"):
        return "release_v6"
    # Default / train -> V1 (the SPEC training split).
    return "release_v1"


def _parse_lcb_tests(row: Any) -> list[dict[str, str]]:
    """Best-effort extraction of LiveCodeBench public test cases.

    LiveCodeBench schemas vary across mirrors. We accept either a JSON-encoded
    string or an already-parsed list under several common keys, and normalise to
    a list of ``{"input": ..., "output": ...}`` dicts. Returns ``[]`` if nothing
    parseable is found (the reward checker treats empty tests as unscoreable).
    """
    import json

    raw = _row_get(row, "public_test_cases", "test_cases", "tests", default=None)
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return []
    if not isinstance(raw, list):
        return []
    tests: list[dict[str, str]] = []
    for case in raw:
        if isinstance(case, dict):
            inp = case.get("input", case.get("stdin", ""))
            out = case.get("output", case.get("expected_output", ""))
            tests.append({"input": str(inp), "output": str(out)})
    return tests


# --------------------------------------------------------------------------- #
# Offline toy fallbacks (no network, for smoke tests S4/S5)
# --------------------------------------------------------------------------- #
def _toy_tasks(benchmark: str) -> list[Task]:
    """Hand-written tiny task set so smoke tests run without ``datasets``/network.

    Each set has 2-3 deterministic, self-contained items whose ``answer`` matches
    the format the corresponding reward checker expects.
    """
    if benchmark == "math500":
        return [
            Task(
                task_id="math500-toy-0",
                benchmark="math500",
                prompt="What is 2 + 2? Give the final answer in \\boxed{}.",
                answer="4",
                meta={"source": "toy"},
            ),
            Task(
                task_id="math500-toy-1",
                benchmark="math500",
                prompt=(
                    "A train travels 60 miles in 1.5 hours. What is its average "
                    "speed in miles per hour? Put the answer in \\boxed{}."
                ),
                answer="40",
                meta={"source": "toy"},
            ),
            Task(
                task_id="math500-toy-2",
                benchmark="math500",
                prompt="Compute 7 * 8. Give the final answer in \\boxed{}.",
                answer="56",
                meta={"source": "toy"},
            ),
        ]
    if benchmark == "mmlu":
        return [
            Task(
                task_id="mmlu-toy-0",
                benchmark="mmlu",
                prompt=_format_mcq(
                    "What is the chemical symbol for water?",
                    ["CO2", "H2O", "O2", "NaCl"],
                ),
                answer="B",
                meta={"source": "toy", "choices": ["CO2", "H2O", "O2", "NaCl"]},
            ),
            Task(
                task_id="mmlu-toy-1",
                benchmark="mmlu",
                prompt=_format_mcq(
                    "Which planet is closest to the Sun?",
                    ["Venus", "Earth", "Mercury", "Mars"],
                ),
                answer="C",
                meta={
                    "source": "toy",
                    "choices": ["Venus", "Earth", "Mercury", "Mars"],
                },
            ),
        ]
    if benchmark == "gpqa":
        return [
            Task(
                task_id="gpqa-toy-0",
                benchmark="gpqa",
                prompt=_format_mcq(
                    "Which fundamental force binds quarks inside a proton?",
                    [
                        "Electromagnetic force",
                        "The strong nuclear force",
                        "Gravity",
                        "The weak nuclear force",
                    ],
                ),
                answer="B",
                meta={"source": "toy"},
            ),
            Task(
                task_id="gpqa-toy-1",
                benchmark="gpqa",
                prompt=_format_mcq(
                    "What is the approximate speed of light in a vacuum?",
                    [
                        "3 x 10^6 m/s",
                        "3 x 10^8 m/s",
                        "3 x 10^10 m/s",
                        "3 x 10^4 m/s",
                    ],
                ),
                answer="B",
                meta={"source": "toy"},
            ),
        ]
    if benchmark == "livecodebench":
        return [
            Task(
                task_id="lcb-toy-0",
                benchmark="livecodebench",
                prompt=(
                    "Read an integer n from standard input and print n * n.\n"
                    "Input: a single integer.\nOutput: the square of the integer."
                ),
                answer={
                    "tests": [
                        {"input": "3\n", "output": "9"},
                        {"input": "5\n", "output": "25"},
                    ],
                    "fn_name": None,
                    "starter_code": None,
                },
                meta={"source": "toy"},
            ),
            Task(
                task_id="lcb-toy-1",
                benchmark="livecodebench",
                prompt=(
                    "Read two integers a and b on one line separated by a space "
                    "and print their sum."
                ),
                answer={
                    "tests": [
                        {"input": "2 3\n", "output": "5"},
                        {"input": "10 -4\n", "output": "6"},
                    ],
                    "fn_name": None,
                    "starter_code": None,
                },
                meta={"source": "toy"},
            ),
        ]
    if benchmark == "gsm8k":
        return [
            Task(
                task_id="gsm8k-toy-0",
                benchmark="gsm8k",
                prompt=(
                    "Natalia sold clips to 12 of her friends in April, and then she sold "
                    "half as many clips in May. How many clips did Natalia sell "
                    "altogether in April and May? Give the final answer in \\boxed{}."
                ),
                answer="18",
                meta={"source": "toy"},
            ),
            Task(
                task_id="gsm8k-toy-1",
                benchmark="gsm8k",
                prompt=(
                    "A robe takes 2 bolts of blue fiber and half that much white fiber. "
                    "How many bolts of fiber does it take in total? "
                    "Give the final answer in \\boxed{}."
                ),
                answer="3",
                meta={"source": "toy"},
            ),
        ]
    if benchmark == "humaneval":
        return [
            Task(
                task_id="HumanEval/toy-0",
                benchmark="humaneval",
                prompt=_format_humaneval_prompt(
                    'def add(a: int, b: int) -> int:\n    """Return the sum of a and b."""\n'
                ),
                answer={
                    "tests": [
                        "def check(candidate):\n"
                        "    assert candidate(2, 3) == 5\n"
                        "    assert candidate(-1, 1) == 0\n"
                        "check(add)\n"
                    ],
                    "fn_name": "add",
                    "starter_code": (
                        'def add(a: int, b: int) -> int:\n    """Return the sum of a and b."""\n'
                    ),
                },
                meta={"source": "toy", "entry_point": "add"},
            ),
            Task(
                task_id="HumanEval/toy-1",
                benchmark="humaneval",
                prompt=_format_humaneval_prompt(
                    'def is_even(n: int) -> bool:\n    """Return True if n is even, else False."""\n'
                ),
                answer={
                    "tests": [
                        "def check(candidate):\n"
                        "    assert candidate(4) is True\n"
                        "    assert candidate(7) is False\n"
                        "check(is_even)\n"
                    ],
                    "fn_name": "is_even",
                    "starter_code": (
                        'def is_even(n: int) -> bool:\n'
                        '    """Return True if n is even, else False."""\n'
                    ),
                },
                meta={"source": "toy", "entry_point": "is_even"},
            ),
        ]
    if benchmark == "bbh":
        return [
            Task(
                task_id="bbh-toy-0",
                benchmark="bbh",
                prompt=_format_bbh_prompt(
                    "Is the following sentence plausible? "
                    '"John threw a touchdown pass."\n(A) plausible\n(B) implausible'
                ),
                answer="(A)",
                meta={"source": "toy", "subtask": "sports_understanding"},
            ),
            Task(
                task_id="bbh-toy-1",
                benchmark="bbh",
                prompt=_format_bbh_prompt(
                    "Sort the following words alphabetically: banana, apple, cherry."
                ),
                answer="apple banana cherry",
                meta={"source": "toy", "subtask": "word_sorting"},
            ),
        ]
    raise ValueError(
        f"Unknown benchmark {benchmark!r}. Supported: {SUPPORTED_BENCHMARKS}"
    )


_HF_LOADERS = {
    "math500": _load_math500_hf,
    "mmlu": _load_mmlu_hf,
    "gpqa": _load_gpqa_hf,
    "livecodebench": _load_livecodebench_hf,
    "gsm8k": _load_gsm8k_hf,
    "humaneval": _load_humaneval_hf,
    "bbh": _load_bbh_hf,
}


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def load_tasks(
    benchmark: str,
    split: str,
    max_items: int | None,
    seed: int = 0,
) -> list[Task]:
    """Load a benchmark as a deterministic list of :class:`Task`.

    Tries the HuggingFace ``datasets`` loader first (lazy/guarded import). If
    ``datasets`` or the network is unavailable -- or the dataset id is gated /
    missing -- it transparently falls back to a tiny built-in toy set so smoke
    tests run offline.

    The returned list is deterministically shuffled by ``seed`` and then
    truncated to ``max_items`` (if not ``None``), so repeated calls with the same
    arguments yield identical results.

    Parameters
    ----------
    benchmark:
        One of :data:`SUPPORTED_BENCHMARKS`.
    split:
        Logical split passed to the loader, e.g. ``"train"`` / ``"test"``. For
        LiveCodeBench this maps to the release version (V1 train / V6 eval).
    max_items:
        Cap on the number of tasks returned; ``None`` means all.
    seed:
        Seed controlling the deterministic shuffle (and toy/HF parity).

    Returns
    -------
    list[Task]
        The (possibly truncated) list of tasks for the benchmark/split.

    Raises
    ------
    ValueError
        If ``benchmark`` is not supported.
    """
    if benchmark not in _HF_LOADERS:
        raise ValueError(
            f"Unknown benchmark {benchmark!r}. Supported: {SUPPORTED_BENCHMARKS}"
        )

    tasks = _HF_LOADERS[benchmark](split)
    if not tasks:
        # Offline / failed load -> built-in toy set.
        tasks = _toy_tasks(benchmark)

    # Deterministic shuffle for reproducible minibatch composition across runs.
    rng = random.Random(seed)
    tasks = list(tasks)
    rng.shuffle(tasks)

    if max_items is not None:
        tasks = tasks[: max(0, int(max_items))]
    return tasks


def sample_minibatch(
    tasks: list[Task],
    m: int,
    rng: random.Random,
) -> list[Task]:
    """Sample ``m`` distinct task instances for one CMA candidate evaluation.

    Per SPEC §5.2 each of the ``m_CMA`` replications uses a different randomly
    sampled task instance (a minibatch of distinct problems per candidate,
    re-sampled per iteration). Sampling is *without replacement* when enough
    tasks exist, otherwise it falls back to sampling *with replacement* so a tiny
    toy set still yields a full minibatch for smoke tests.

    Parameters
    ----------
    tasks:
        The pool of tasks to draw from (typically the training split).
    m:
        Number of instances to draw (``m_CMA``, e.g. 16).
    rng:
        Caller-owned :class:`random.Random` so the optimizer controls determinism
        (e.g. re-seeded per CMA iteration).

    Returns
    -------
    list[Task]
        ``m`` sampled tasks (distinct where possible).

    Raises
    ------
    ValueError
        If ``tasks`` is empty or ``m`` is not positive.
    """
    if not tasks:
        raise ValueError("Cannot sample a minibatch from an empty task list.")
    if m <= 0:
        raise ValueError(f"Minibatch size m must be positive, got {m}.")

    if m <= len(tasks):
        return rng.sample(tasks, m)
    # Not enough distinct tasks (toy set): sample with replacement.
    return [rng.choice(tasks) for _ in range(m)]
