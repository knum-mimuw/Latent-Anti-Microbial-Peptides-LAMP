# pep-eval

Standalone AMP challenge evaluation package.

## Purpose

- Load peptide datasets from Hugging Face (hub or disk)
- Compute AMP challenge category/tie-break metrics
- Log metrics, tags, and structured artifacts directly to MLflow

## Package layout

- `src/pep_eval/panels.py`: canonical strain panels
- `src/pep_eval/parsing.py`: assay value parsing
- `src/pep_eval/io.py`: dataset loading
- `src/pep_eval/metrics.py`: per-peptide and aggregate metrics
- `src/pep_eval/logging.py`: MLflow logging utilities
- `src/pep_eval/api.py`: orchestration entrypoint used by pipelines
