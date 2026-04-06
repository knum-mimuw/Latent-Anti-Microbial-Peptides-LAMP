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

3. Create a per-run config file:

   ```bash
   cp src/pipelines/pipelines/configs/run/grugruval.yaml /tmp/lamp-run.yaml
   ```

## Usage

### Standalone (no ZenML)

Training works without ZenML -- just use the Lightning CLI directly. Pass explicit
`--config` files for trainer, model, data, and any logger/callbacks (nothing is
auto-discovered). From the **repository root**:

```bash
uv run modelling fit \
  --config src/modelling/configs/trainer/grugru_vae.yaml \
  --config src/modelling/configs/model/grugru_vae.yaml \
  --config src/modelling/configs/data/grugru_vae.yaml \
  --config src/modelling/configs/logger/mlflow_local.yaml \
  --config src/modelling/configs/callbacks/checkpoint.yaml
```

See `src/modelling/README.md` for the same command using paths under `src/modelling/`.

### Via ZenML Pipeline

The same training, orchestrated by ZenML for lineage and metadata while MLflow
remains the canonical owner of checkpoints and metrics:

```bash
uv run python -m pipelines.training \
  /tmp/lamp-run.yaml \
  src/modelling/configs/trainer/grugru_vae.yaml \
  src/modelling/configs/model/grugru_vae.yaml \
  src/modelling/configs/data/grugru_vae.yaml \
  src/modelling/configs/logger/mlflow_local.yaml \
  src/modelling/configs/callbacks/checkpoint.yaml
```

This flow writes a deterministic `training_manifest.json` during training and
logs both the checkpoint directory and manifest into the MLflow run.

### Publish From MLflow To Hugging Face

Use the standalone publish pipeline when you want to release a checkpoint from a
known MLflow run and artifact path:

```bash
uv run python -m pipelines.publish_hf \
  /tmp/lamp-run.yaml \
  --run-id 8dcb3c4c7d4a4f54a58fd52ef0a5ef12 \
  --artifact-path checkpoints/epoch=30-step=1200-val/loss=0.1234.ckpt \
  --tag run-20260405
```

Or publish from a local training manifest that already contains those explicit
coordinates:

```bash
uv run python -m pipelines.publish_hf \
  /tmp/lamp-run.yaml \
  --manifest-path /tmp/training_manifest.json \
  --tag run-20260405
```

This release upload includes:
- Hub-native pretrained weights/config
- The custom model source files required for `from_pretrained(..., trust_remote_code=True)`
- A generated `README.md` model card
- Structured metadata in `lamp_metadata.json`

### Train And Optionally Publish

Use the composed pipeline when you want training to automatically publish the
best checkpoint after it finishes:

```bash
uv run python -m pipelines.train_and_publish \
  --run-config /tmp/lamp-run.yaml \
  src/modelling/configs/trainer/grugru_vae.yaml \
  src/modelling/configs/model/grugru_vae.yaml \
  src/modelling/configs/data/grugru_vae.yaml \
  src/modelling/configs/logger/mlflow_local.yaml \
  src/modelling/configs/callbacks/checkpoint.yaml \
  --upload-to-hf \
  --tag run-20260405
```

Leave `--upload-to-hf` off to keep this as a pure train-and-log run.

### Re-Upload After Fine-Tuning

The recommended flow is one stable HF repo per model family, with each new model
state uploaded as a new revision or tag:

```bash
uv run python -m pipelines.publish_hf \
  /tmp/lamp-run.yaml \
  --run-id c138b2d5e34c4d2da4ed1ebc4c02ab77 \
  --artifact-path checkpoints/finetuned-01.ckpt \
  --tag finetune-01
```

Consumers can keep loading from the same repo and pin a specific release with
`revision="finetune-01"`.

## Environment

Set these in `.env` / direnv only for environment-level concerns:

- `HF_TOKEN` -- write token for pushing model artifacts to Hugging Face
- `MLFLOW_CHECKPOINT_ARTIFACT_PATH` -- canonical MLflow artifact subtree for checkpoints
- `MLFLOW_MANIFEST_ARTIFACT_PATH` -- canonical MLflow artifact subtree for manifests
- `TRAINING_MANIFEST_PATH` -- optional local override for where the training manifest is written

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

The pipelines package contains **thin wrappers** that call into the modelling
package's public API or CLI. The modelling package has **zero ZenML imports**.

```
src/pipelines/                    # workspace member (depends on zenml + lamp-modelling)
├── pyproject.toml
└── pipelines/                    # import package ``pipelines``
    ├── training.py             # @pipeline: training_pipeline
    ├── publish_hf.py           # @pipeline: publish_hf_pipeline
    ├── train_and_publish.py    # @pipeline: train_and_optional_publish_pipeline
    ├── configs/run/grugruval.yaml
    └── steps/
        ├── train_step.py       # @step: runs training and reads the manifest
        └── publish_hf_step.py  # @step: exports MLflow artifacts to HF
```

## Optional: MLflow Server

For a richer MLflow UI experience, start a local server:

```bash
./scripts/start_mlflow_server.sh
```

Then set `MLFLOW_TRACKING_URI=http://127.0.0.1:5000` in your `.env`.
File-based mode (`file:./mlflow-store`) works without any server.
