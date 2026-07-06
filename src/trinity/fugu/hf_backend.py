"""HF trainable backend for the Fugu Conductor policy.

This module is imported only on the GPU box. It implements the
``PolicyBackend`` protocol from :mod:`trinity.fugu.grpo` without depending on
TRL: rollouts are sampled with ``generate`` and the update recomputes token
log-probs for the emitted workflow under the current policy, weighted by the
group-normalized GRPO advantages.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from trinity.fugu.conductor import Proposal, build_prompt
from trinity.fugu.workflow import MAX_STEPS
from trinity.types import Task

__all__ = ["HFBackendConfig", "HFPolicyBackend"]

_DEFAULT_SUBTASK = "Solve the problem and end with the final answer in the required format."


@dataclass
class HFBackendConfig:
    model_name: str = "Qwen/Qwen3-0.6B"
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    lr: float = 1e-6
    max_new_tokens: int = 512
    max_prompt_tokens: int = 4096
    sample_temperature: float = 1.0
    greedy_temperature: float = 0.0
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    gradient_accumulation: int = 1
    proposal_prefix: str = "model_id = ["
    # Constrained decoding: structurally guarantee a schema-valid proposal so the
    # parse-gate cannot reject it. The policy still chooses the routing (how many
    # steps, which worker per step); subtasks/access_list are assembled canonically.
    constrained: bool = False
    constrained_allow_self: bool = False  # keep in sync with max_depth > 0
    constrained_max_steps: int = MAX_STEPS


class HFPolicyBackend:
    """Trainable local Conductor backed by ``transformers``.

    The backend knows the worker menu so ``update`` can reconstruct the exact
    prompt for every stored rollout from ``GroupResult.task`` +
    ``WorkflowRun.raw_proposal``. This keeps the generic GRPO loop free of torch.
    """

    def __init__(self, cfg: HFBackendConfig, worker_names: list[str]):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg
        self.worker_names = list(worker_names)
        self.torch = torch
        self.device = torch.device(cfg.device)
        self.dtype = _torch_dtype(torch, cfg.dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        ).to(self.device)
        self.model.train()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self._updates = 0

    async def propose(
        self,
        task: Task,
        worker_names: list[str],
        *,
        sample: bool = False,
        rng=None,
        client=None,
    ) -> Proposal:
        del rng, client
        if list(worker_names) != self.worker_names:
            self.worker_names = list(worker_names)

        if self.cfg.constrained:
            return self._propose_constrained(task, sample=sample)

        torch = self.torch
        prompt = self._prompt_text(task)
        enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.cfg.max_prompt_tokens,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.cfg.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if sample:
            gen_kwargs.update(
                do_sample=True,
                temperature=max(1e-5, self.cfg.sample_temperature),
                top_p=0.95,
            )
        else:
            gen_kwargs.update(do_sample=False)

        self.model.eval()
        with torch.inference_mode():
            out = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )
        self.model.train()

        prompt_len = int(input_ids.shape[1])
        gen_ids = out[0, prompt_len:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        text = f"{self.cfg.proposal_prefix}{text}".strip()
        return Proposal(
            text=text,
            prompt_tokens=prompt_len,
            completion_tokens=int(gen_ids.numel()),
        )

    # ------------------------------------------------------------------ #
    # Constrained decoding
    # ------------------------------------------------------------------ #
    def _propose_constrained(self, task: Task, *, sample: bool) -> Proposal:
        """Emit a workflow that is schema-valid by construction.

        The policy genuinely chooses the routing: a constrained decode samples
        the per-step worker index (and the number of steps) from the model's own
        next-token distribution, restricted to the legal worker ids and the
        list-continue/close tokens. The subtask strings and the access DAG are
        then assembled canonically (always non-empty, always a valid DAG), so the
        resulting proposal always passes ``parse_workflow``. Parse-validity comes
        from the canonical assembly; the constrained sample only supplies routing
        diversity (so two rollouts can route differently and produce real GRPO
        advantage).
        """
        prompt = self._prompt_text(task)  # ends with the "model_id = [" prefix
        enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.cfg.max_prompt_tokens,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        prompt_len = int(input_ids.shape[1])

        model_ids, n_gen = self._constrained_int_list(
            input_ids, attention_mask, sample=sample
        )
        n_workers = len(self.worker_names)
        hi = n_workers if self.cfg.constrained_allow_self else max(0, n_workers - 1)
        max_steps = max(1, min(int(self.cfg.constrained_max_steps), MAX_STEPS))
        clamped = [min(max(0, int(m)), hi) for m in model_ids][:max_steps]
        if not clamped:
            clamped = [0]

        text = _canonical_workflow(clamped)
        return Proposal(
            text=text,
            prompt_tokens=prompt_len,
            completion_tokens=int(n_gen),
        )

    def _constrained_int_list(self, input_ids, attention_mask, *, sample: bool):
        """Constrained decode of the ``model_id`` list: ints then ``,``/``]``.

        Returns ``(worker_ids, n_generated_tokens)``. Falls back to ``[0]`` if the
        tokenizer cannot supply clean single-token digits/brackets (the canonical
        assembly still guarantees a parseable proposal in that case).
        """
        torch = self.torch
        n_workers = len(self.worker_names)
        hi = n_workers if self.cfg.constrained_allow_self else max(0, n_workers - 1)
        max_steps = max(1, min(int(self.cfg.constrained_max_steps), MAX_STEPS))

        digit_ids: dict[int, int] = {}
        for d in range(0, hi + 1):
            tid = self._single_token_id(str(d))
            if tid is not None:
                digit_ids[tid] = d
        comma_id = self._single_token_id(",")
        close_id = self._single_token_id("]")
        if not digit_ids or close_id is None:
            return [0], 0

        cur_ids = input_ids
        cur_mask = attention_mask
        out: list[int] = []
        state = "digit"
        n_gen = 0
        gen_cap = max_steps * 3 + 2

        self.model.eval()
        with torch.inference_mode():
            while n_gen < gen_cap:
                logits = self.model(
                    input_ids=cur_ids, attention_mask=cur_mask
                ).logits[:, -1, :]
                if state == "digit":
                    allowed = list(digit_ids.keys())
                else:
                    allowed = [close_id]
                    if comma_id is not None and len(out) < max_steps:
                        allowed.insert(0, comma_id)
                tok = self._pick_from(logits, allowed, sample)
                cur_ids = torch.cat([cur_ids, tok.view(1, 1)], dim=1)
                cur_mask = torch.cat(
                    [cur_mask, torch.ones((1, 1), dtype=cur_mask.dtype, device=cur_mask.device)],
                    dim=1,
                )
                n_gen += 1
                tok_id = int(tok.item())
                if state == "digit":
                    out.append(digit_ids[tok_id])
                    state = "sep"
                else:
                    if tok_id == close_id:
                        break
                    state = "digit"
        self.model.train()
        return (out or [0]), n_gen

    def _pick_from(self, logits, allowed_ids: list[int], sample: bool):
        """Pick one token from ``allowed_ids`` by masking all other logits to -inf."""
        torch = self.torch
        row = logits[0]
        idx = torch.tensor(allowed_ids, device=row.device, dtype=torch.long)
        masked = torch.full_like(row, float("-inf"))
        masked[idx] = row[idx]
        if sample:
            temp = max(1e-5, self.cfg.sample_temperature)
            probs = torch.softmax(masked / temp, dim=-1)
            return torch.multinomial(probs, num_samples=1)[0]
        return torch.argmax(masked)

    def _single_token_id(self, s: str):
        """Token id for ``s`` iff it encodes to exactly one token, else ``None``."""
        ids = self.tokenizer.encode(s, add_special_tokens=False)
        return ids[0] if len(ids) == 1 else None

    def update(self, groups: list) -> dict:
        """Apply one GRPO policy-gradient update.

        Loss per sample is ``advantage * token_nll``. Positive-advantage
        workflows are made more likely; negative-advantage workflows less likely.
        Zero-advantage samples are skipped because they carry no group signal.
        """
        torch = self.torch
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        n_samples = 0
        n_tokens = 0
        loss_sum = 0.0
        abs_adv_sum = 0.0
        accum = max(1, int(self.cfg.gradient_accumulation))

        for group in groups:
            for run, adv in zip(group.runs, group.advantages):
                if abs(float(adv)) < 1e-8 or not run.raw_proposal:
                    continue
                loss, toks = self._sample_nll(group.task, run.raw_proposal)
                weighted = loss * float(adv) / accum
                weighted.backward()
                loss_sum += float(weighted.detach().cpu()) * accum
                abs_adv_sum += abs(float(adv))
                n_samples += 1
                n_tokens += toks
                if n_samples % accum == 0:
                    if self.cfg.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

        if n_samples and n_samples % accum:
            if self.cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        self._updates += 1
        return {
            "backend": "hf",
            "updates": self._updates,
            "samples": n_samples,
            "tokens": n_tokens,
            "mean_weighted_loss": loss_sum / max(1, n_samples),
            "mean_abs_advantage": abs_adv_sum / max(1, n_samples),
        }

    def format_warmup(
        self,
        tasks: list[Task],
        *,
        steps: int = 20,
        batch_size: int = 1,
        model_id: int = 0,
    ) -> dict:
        """Supervised grammar warmup on synthetic parseable workflows.

        This is deliberately format-only: the target is a generic one-step
        workflow that delegates solving to a worker. It does not teach task
        answers or use paid worker calls; it only makes GRPO rollouts more
        likely to pass the parser so rewards can start flowing.
        """
        torch = self.torch
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        if not tasks or steps <= 0:
            return {"steps": 0, "examples": 0, "mean_loss": 0.0}

        losses: list[float] = []
        examples = 0
        accum = max(1, int(batch_size))
        for step in range(steps):
            task = tasks[step % len(tasks)]
            proposal = _format_warmup_target(model_id)
            loss, _ = self._sample_nll(task, proposal)
            (loss / accum).backward()
            losses.append(float(loss.detach().cpu()))
            examples += 1
            if examples % accum == 0:
                if self.cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

        if examples % accum:
            if self.cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        return {
            "steps": steps,
            "examples": examples,
            "mean_loss": sum(losses) / max(1, len(losses)),
            "target_model_id": model_id,
        }

    def save_pretrained(self, path: str) -> None:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def _sample_nll(self, task: Task, proposal: str):
        torch = self.torch
        prompt = self._prompt_text(task)
        full = self._assistant_text(task, proposal)
        prompt_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.cfg.max_prompt_tokens,
        )["input_ids"][0]
        enc = self.tokenizer(
            full,
            return_tensors="pt",
            truncation=True,
            max_length=self.cfg.max_prompt_tokens + self.cfg.max_new_tokens,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        labels = input_ids.clone()
        prompt_len = min(int(prompt_ids.numel()), int(labels.shape[1]))
        labels[:, :prompt_len] = -100
        valid = labels.ne(-100).sum().clamp_min(1)
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        # HF returns mean CE over unmasked labels. Keep it length-normalized so a
        # long parseable workflow is not punished solely for having more tokens.
        return out.loss.to(torch.float32), int(valid.item())

    def _prompt_text(self, task: Task) -> str:
        messages = build_prompt(task, self.worker_names)
        return (
            _chat_text(self.tokenizer, messages, add_generation_prompt=True)
            + self.cfg.proposal_prefix
        )

    def _assistant_text(self, task: Task, proposal: str) -> str:
        if proposal.startswith(self.cfg.proposal_prefix):
            proposal = proposal[len(self.cfg.proposal_prefix):]
        return self._prompt_text(task) + proposal


def _chat_text(tokenizer, messages: list[dict], *, add_generation_prompt: bool) -> str:
    if getattr(tokenizer, "chat_template", None):
        kwargs = dict(
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        try:
            return tokenizer.apply_chat_template(
                messages,
                enable_thinking=False,
                **kwargs,
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, **kwargs)
    chunks = []
    for m in messages:
        role = m.get("role", "user").upper()
        chunks.append(f"{role}:\n{m.get('content', '')}")
    if add_generation_prompt:
        chunks.append("ASSISTANT:\n")
    return "\n\n".join(chunks)


def _torch_dtype(torch, name: str):
    key = (name or "").lower()
    if key in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if key in {"fp16", "float16", "half"}:
        return torch.float16
    if key in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _format_warmup_target(model_id: int) -> str:
    return _canonical_workflow([int(model_id)])


def _canonical_workflow(model_ids: list[int]) -> str:
    """Assemble a guaranteed parse-valid workflow from a list of worker indices.

    Each step gets the canonical subtask; the access DAG reads only the query for
    every step except the final one, which reads ``"all"`` prior outputs to
    synthesize (always a valid DAG since it points strictly backward). A
    single-step workflow uses ``[]`` (query only). The result always satisfies
    :func:`trinity.fugu.workflow.parse_workflow` for any non-empty ``model_ids``.
    """
    ids = list(model_ids) or [0]
    n = len(ids)
    models = ", ".join(str(int(m)) for m in ids)
    import json as _json

    subs = ", ".join(_json.dumps(_DEFAULT_SUBTASK) for _ in range(n))
    if n == 1:
        access = "[]"
    else:
        access = ", ".join(["[]"] * (n - 1) + ['"all"'])
    return f"model_id = [{models}]\nsubtasks = [{subs}]\naccess_list = [{access}]"
