# Minirouter Contribution Guide

This repository is used by multiple miners. Each miner must work in a dedicated branch and submit changes through a pull request.

## Branch naming

Use this exact branch prefix format:

```text
sn74-<miner-name>
```

Examples:

- `sn74-tmimmanuel`
- `sn74-alice`
- `sn74-bob`

Do not use the `main` branch for active work.

## Required workflow

1. Fork the upstream repository to your own GitHub account.
2. Clone your fork locally.
3. Create your working branch from `main` using the required prefix.
4. Do all work in that branch only.
5. Push the branch to your fork.
6. Open a pull request from your branch back to the upstream repo, or to your assigned integration branch if the competition server requires that.

## Example

```bash
git checkout main
git pull upstream main
git checkout -b sn74-tmimmanuel
```

If your fork is the default remote:

```bash
git push origin sn74-tmimmanuel
```

## What to submit

Final submissions should include the trained model artifact and the evaluation metadata in `submissions/final_model/`.

Expected files:

- `best_theta.npy`
- `summary.json`
- `history.json` if you want to include training history
- `eval.json` if you want to include a local evaluation report

Before opening a PR, make sure the model bundle is complete and the local eval command succeeds.
You can check the bundle offline (no API keys, no GPU) with:

```bash
python scripts/validate_submission.py            # defaults to submissions/final_model/
python scripts/validate_submission.py --dir path/to/bundle
```

It verifies the required files are present, that `best_theta.npy` is a finite float
vector of the expected length, and that `summary.json` is valid — exiting non-zero on
any problem so you catch mistakes before the validator backend does.

## Rules

- Keep each miner's work isolated to their branch.
- Do not mix unrelated miners' changes in the same branch.
- Rebase or merge from upstream as needed before submitting.
- Include a clear commit message and PR title.
- Never commit secrets.

## Recommended PR content

- What changed
- Which benchmark or setup was used
- The final `best_theta.npy`
- The evaluation score and benchmark settings
- Any caveats or known limits

