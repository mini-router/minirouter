<!--
Submission PR template.
Use this only for final-submission PRs that include
submissions/final_model/best_theta.npy (branch prefix sn74-<miner-name>).
For code/docs/infra PRs, use the default pull request template instead.
-->

## Miner

- Branch: `sn74-<miner-name>`
- Miner name:

## What changed

<!-- Summary of the approach, training run, or routing change behind this submission -->

## Benchmark & setup

- Benchmark:
- Provider:
- Models config:

## MiniBridge identity used for evaluation

<!--
owner_id / key_id identify which MiniBridge-registered key this submission's
evaluation calls were billed against (NOT the API key itself — MiniBridge
never exposes the underlying key). Fill these in so reviewers can trace which
credential/spend-limit bucket this submission's eval run used.
-->

- `owner_id`:
- `key_id`:

## Final model bundle

- [ ] `submissions/final_model/best_theta.npy` included
- [ ] `submissions/final_model/summary.json` included
- [ ] `python utility/validate_submission.py --dir submissions/final_model` passes locally

## Evaluation score

- Local eval score:
- Benchmark settings used to obtain it:

## Caveats or known limits

<!-- Anything a reviewer should know before promoting this submission -->
