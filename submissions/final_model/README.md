This folder is the final submit-ready model bundle for king / submission PRs.

Required:

- `best_theta.npy` — flat float64 vector of length 13,312 (head 6,144 + SVF 7,168)
- `summary.json` — training / ship metadata

Optional:

- `history.json`
- `eval.json`

Ship policy (issue #234): `best_theta.npy` must be the CMA-ES distribution mean
(`xfavorite`), not the luckiest noisy evaluation (`xbest`).

Validate offline before opening a PR:

```bash
python utility/validate_submission.py --dir submissions/final_model
```
