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
The PR will open as `awaiting_ci`; a maintainer starts the GitHub Actions `PR automation`
workflow manually when the submission is ready to queue on the validator backend.

Validate the bundle offline first (no API keys required):

```bash
python utility/validate_submission.py --dir submissions/final_model
```

The checker confirms required files exist, `best_theta.npy` has the expected length
(13,312 by default), and `summary.json` is valid JSON. Fix any reported errors before
pushing.

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
