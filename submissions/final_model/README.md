This folder is for the final submit-ready model bundle.

Place these files here before packaging a submission:

- `best_theta.npy`
- `summary.json`
- `history.json` or `eval.json` if available

Then run the offline checker:

```bash
python scripts/validate_submission.py --dir submissions/final_model
```
