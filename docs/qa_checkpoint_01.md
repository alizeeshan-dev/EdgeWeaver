# QA Checkpoint 01 — Phases 1–2

## Intended scope and implementation

Phases 1–2 provide validated domain/configuration models, the deterministic five-request SimPy
fixture, official UCI HAR ingestion with its predefined split, three reproducible scikit-learn
training pipelines, saved preprocessors/estimators, evaluation metadata, artifact reload/prediction,
and real local single-request latency profiling. Main code is in `edgeweaver/domain.py`,
`edgeweaver/config.py`, `edgeweaver/simulation.py`, and `edgeweaver/ml/`; wrappers are in `scripts/`.
Configurations, model metadata, profiles, raw timings, and environment metadata are under
`configs/` and `artifacts/`.

Expected behavior: official data validates as train `(7352, 561)` and test `(2947, 561)`; seed 2027
reproduces preprocessing/model predictions and profiling sample indices; real timing values remain
nondeterministic. Saved artifacts reload and predict with the exact fitted preprocessor. The Phase 1
fixture remains exactly 20 ordered events, five completions, and three met deadlines. Synthetic and
measured profiles remain distinguished by `profile_kind`.

## Useful QA commands

```bash
python -m pip install -e ".[dev]"
python scripts/download_uci_har.py
python scripts/train_models.py --seed 2027
python scripts/verify_models.py --samples 32
python scripts/profile_models.py --predictions 500 --warmups 20 --seed 2027
python -m edgeweaver
pytest
ruff check edgeweaver scripts tests
ruff format --check edgeweaver scripts tests
mypy edgeweaver
```

## Invariants and risk areas

- Never concatenate or re-split UCI train/test data; labels and subjects must align with feature rows.
- Scalers fit only on training data; random forest uses a saved identity transformer.
- IDs map `light` → logistic regression, `balanced` → random forest, `heavy` → MLP; role names do not
  assert an accuracy or latency ordering.
- Profile timing includes one-row preprocessing plus prediction, excludes loading/warm-up, retains
  raw nanoseconds, uses population std and linear P95, and requires at least 500 observations.
- Model size means serialized estimator bytes only; preprocessor size is recorded separately.
- High-risk areas are malformed/nested ZIP extraction, joblib compatibility across library versions,
  path/size metadata drift, Windows timing noise, and future accidental use of synthetic values.
- Worth testing further: unsafe ZIP paths, corrupt archives, NaN/unknown-label data, feature-count
  mismatch at inference, interrupted downloads, read-only artifact destinations, and multiple reruns.

Assumptions: the current official UCI archive wraps an inner `UCI HAR Dataset.zip`; the downloader
supports both nested and direct layouts. Generated raw data and joblib binaries are intentionally
git-ignored but are present locally. Only this physical computer was profiled.

Incomplete/deferred by design: no simulation expansion, network execution, scheduler policies,
workload scenarios, experiment runner, analysis/charts, FastAPI, React, deployment, authentication,
database, production queues, or QA-driven speculative refactors should exist yet.
