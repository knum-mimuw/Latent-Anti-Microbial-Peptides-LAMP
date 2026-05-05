# Pipelines Package

ZenML pipeline orchestration for LAMP training workflows. This is an **optional
layer** -- all operations can be performed standalone via the `modelling` CLI.

## Prerequisites

1. Install the workspace with ZenML dependencies:

   ```bash
   uv sync
   ```

2. Initialize the ZenML stack (one-time setup):

   ```bash
   ./scripts/init_zenml_stack.sh
   ```

   This registers a local stack named `lamp-local` with:
   - Local orchestrator
   - Local artifact store (`zenml-artifacts/`)
   - MLflow experiment tracker (reads `MLFLOW_TRACKING_URI` from environment)

3. Copy per-run YAML configs from templates:

   **Training only** (MLflow experiment name):

   ```bash
   cp src/pipelines/pipelines/configs/run/training/grugru_vae_training_run.yaml /tmp/lamp-train-run.yaml
   ```

   **Publish / train-and-publish with HF** (same MLflow experiment plus ZenML + Hugging Face):

   ```bash
   cp src/pipelines/pipelines/configs/run/training/grugru_vae_hf_publish_run.yaml /tmp/lamp-publish-run.yaml
   ```

   Copy whichever files you need; editing `/tmp/…` copies avoids churn in git-tracked templates.

## Usage

### Standalone (no ZenML)

Training uses Hydra + Hugging Face `Trainer`. From the **repository root**:

```bash
uv run modelling hydra.job.chdir=false
```

Optional YAML fragments (merged as Hydra dotlist overrides) can be passed when using ZenML;
for manual runs, prefer CLI overrides or `--config-name`.

See `src/modelling/README.md`.

### Via ZenML Pipeline

The same training, orchestrated by ZenML for lineage and metadata while MLflow
remains the canonical owner of checkpoints and metrics:

```bash
uv run python -m pipelines.training /tmp/lamp-train-run.yaml
```

Optional Hydra override YAML paths **after** the run config:

```bash
uv run python -m pipelines.training /tmp/lamp-train-run.yaml /tmp/trainOverrides.yaml
```

The `train` step runs HF Trainer via subprocess, then queries MLflow for the
latest finished run and its checkpoint artifacts.

### Publish From MLflow To Hugging Face

Use the standalone publish pipeline when you want to release a checkpoint from a
known MLflow run and artifact path:

```bash
uv run python -m pipelines.publish \
  /tmp/lamp-publish-run.yaml \
  --run-id 8dcb3c4c7d4a4f54a58fd52ef0a5ef12 \
  --artifact-path checkpoints/checkpoint-1200 \
  --tag run-20260405
```

This release upload includes:
- Hub-native pretrained weights/config
- The custom model source files required for `from_pretrained(..., trust_remote_code=True)`
- A generated `README.md` model card
- Structured metadata in `lamp_metadata.json`

### Train And Optionally Publish

Use **`grugru_vae_hf_publish_run.yaml`** (or your `/tmp/lamp-publish-run.yaml` copy) for `--run-config` whenever ZenML needs `repo_id`, tags, or ZenML model linkage.

Pure train-only ZenML runs still pass **`grugru_vae_training_run.yaml`** as the first argument (training pipeline ignores HF/ZenML sections anyway).

```bash
uv run python -m pipelines.training.train_and_publish \
  src/modelling/configs/grugru_vae.yaml \
  --run-config /tmp/lamp-publish-run.yaml \
  --upload-to-hf \
  --tag run-20260405
```

Append extra modelling YAML fragments **before** `--run-config` if your workspace splits configs (`model/*.yaml`, `data/*.yaml`, etc.).

### Evaluate AMP Dataset From Hugging Face

Run the evaluation pipeline to compute all competition categories and log to
MLflow only (this workflow does not write local result files):

```bash
uv run python -m pipelines.evaluation \
  src/pipelines/pipelines/configs/run/evaluation/amp_eval.yaml
```

Or with Taskfile:

```bash
task pipeline:evaluate:amp RUN_CONFIG=src/pipelines/pipelines/configs/run/evaluation/amp_eval.yaml
```

The run config controls dataset source/name/split/revision, column mappings,
and experiment name. Each run is tagged with `hf_dataset_name` and
`hf_dataset_split`, and the run name includes the dataset identifier to support
MLflow ranking/filtering by category metrics.

### Re-Upload After Fine-Tuning

The recommended flow is one stable HF repo per model family, with each new model
state uploaded as a new revision or tag:

```bash
uv run python -m pipelines.publish \
  /tmp/lamp-publish-run.yaml \
  --run-id c138b2d5e34c4d2da4ed1ebc4c02ab77 \
  --artifact-path checkpoints/checkpoint-5000 \
  --tag finetune-01
```

Consumers can keep loading from the same repo and pin a specific release with
`revision="finetune-01"`.

## Environment

Set these in `.env` / direnv only for environment-level concerns:

- `HF_TOKEN` -- write token for pushing model artifacts to Hugging Face
- `MLFLOW_EXPERIMENT_NAME` -- optional default experiment (ZenML `train` sets this from the run YAML)
- `MLFLOW_CHECKPOINT_ARTIFACT_PATH` -- canonical MLflow artifact subtree for checkpoints

Run-varying identities come from a per-run YAML config, not env vars:

```yaml
mlflow:
  experiment_name: lamp-grugru-vae-dev

zenml:
  model_name: lamp-grugru-vae
  model_version: run-20260405

huggingface:
  repo_id: your-org/lamp-grugru-vae
  revision: release/run-20260405
  tag: run-20260405
  model_card_title: LAMP GRU-VAE
  private: false
```

This keeps parallel runs isolated from each other.

## Loading From Hugging Face

Uploaded models include the custom Python files needed for Hub loading:

```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "your-org/lamp-grugru-vae",
    revision="run-20260405",
    trust_remote_code=True,
)
```

## Architecture

The pipelines package contains **thin wrappers** that call into standalone
packages. Training calls `lamp-modelling`; evaluation calls `pep-eval`.

```
src/pipelines/                    # workspace member (depends on zenml + lamp-modelling)
├── pyproject.toml
└── pipelines/                    # import package ``pipelines``
    ├── publish/                # publish domain
    │   ├── __main__.py         # python -m pipelines.publish
    │   ├── pipeline.py         # @pipeline: publish_hf_pipeline
    │   └── steps/
    ├── training/               # training domain
    │   ├── __main__.py         # python -m pipelines.training
    │   ├── pipeline.py         # @pipeline: training_pipeline
    │   ├── train_and_publish.py# python -m pipelines.training.train_and_publish
    │   └── steps/
    ├── evaluation/             # evaluation domain
    │   ├── __main__.py         # python -m pipelines.evaluation
    │   ├── pipeline.py         # @pipeline: evaluation_pipeline
    │   └── steps/
    └── configs/run/
        ├── training/grugru_vae_training_run.yaml       # MLflow experiment (training-only)
        ├── training/grugru_vae_hf_publish_run.yaml     # MLflow + ZenML + Hugging Face (publish / upload)
        └── evaluation/amp_eval.yaml

src/pep_eval/                    # standalone AMP evaluation package
├── pyproject.toml
└── src/pep_eval/
    ├── api.py
    ├── io.py
    ├── metrics.py
    ├── parsing.py
    ├── panels.py
    └── logging.py
```

## Optional: MLflow Server

For a richer MLflow UI experience, start a local server:

```bash
./scripts/start_mlflow_server.sh
```

Then set `MLFLOW_TRACKING_URI=http://127.0.0.1:5000` in your `.env`.
Local SQLite mode (`sqlite:///mlflow-store/mlflow.db`) works without any server.
