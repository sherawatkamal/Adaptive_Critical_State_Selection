# Pilot suite (v8)

These pilots are intended to *validate plumbing*, not to produce publishable numbers.

If these pilots pass, you should be able to scale to the full experiment budgets with confidence.

## Quick start

```bash
# Requires an OpenAI key for teacher hints
export OPENAI_API_KEY=...

python scripts/pilots/run_all_pilots_v8.py \
  --out-dir results/pilot_v8 \
  --base-model sshleifer/tiny-gpt2 \
  --teacher-model gpt-4o-mini \
  --mock-env
```

To run against the real WebShop env (recommended once the mock pilot works):

```bash
python scripts/pilots/run_all_pilots_v8.py \
  --out-dir results/pilot_v8_real \
  --base-model <your-real-student-model> \
  --teacher-model gpt-4o-mini
```

## What it runs

- **Phase 0**: collect rollouts (student)
- **Phase 0**: mine snapshots from rollouts
- **Phase 1**: label snapshots (uncertainty, leverage, CPT, teacher hints)
- **Phase 2**: train a small subset of the training matrix (demo/contrast/hint + baseline)
- **Phase 3**: run end-to-end and snapshot-based evaluation
- **Phase 4**: train the teachability predictor

All outputs are written into the `--out-dir` you provide.
