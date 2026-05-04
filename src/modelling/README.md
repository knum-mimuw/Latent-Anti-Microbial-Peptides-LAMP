# Modelling Package

Hugging Face `transformers.Trainer` training with Hydra-composed configs. Models are
`PreTrainedModel` implementations whose `forward` returns a `loss` when `labels` are
passed (as produced by `TokenizerCollate`).

## Running training

From the **repository root**:

```bash
uv run modelling hydra.job.chdir=false
```

Defaults compose `configs/config.yaml` (model + data + training groups). Switch experiments:

```bash
uv run modelling --config-name=grugru_vae_streaming hydra.job.chdir=false
```

Hydra overrides (examples):

```bash
uv run modelling hydra.job.chdir=false training.num_train_epochs=2 training.output_dir=/tmp/lamp-out
```

Config root on disk: `src/modelling/configs/` (`config.yaml`, `model/`, `data/`, `training/`, and optional top-level experiment YAMLs such as `grugru_vae_streaming.yaml`).

## MLflow

Set in the environment (see repository `.env-default`):

- `MLFLOW_TRACKING_URI` — tracking backend
- `MLFLOW_EXPERIMENT_NAME` — experiment name (also set automatically by ZenML `train` when applicable)

Training uses `TrainingArguments.report_to: [mlflow]`. When a run is active, `ManifestCallback`
writes `training_manifest.json` (path from `TRAINING_MANIFEST_PATH`) and logs checkpoint + manifest
artifacts (`MLFLOW_CHECKPOINT_ARTIFACT_PATH`, `MLFLOW_MANIFEST_ARTIFACT_PATH`).

## Export to Hugging Face Hub

Trainer checkpoints are **directories** (`config.json` + `model.safetensors`, …). Publish:

```bash
uv run python -m modelling.src.utils.export_to_hf \
  --weights-dir /path/to/checkpoint-XXXX \
  --repo-id your-org/your-model
```

(`--checkpoint` is a deprecated alias for `--weights-dir`.)

## Package layout

```
src/modelling/
├── configs/              # Hydra YAML (config.yaml, model/, data/, training/)
├── src/
│   ├── training/         # Hydra entrypoint, build_trainer, dataset helpers
│   ├── callbacks/        # ManifestCallback, IterableEpochCallback
│   ├── datamodules/      # TokenizerCollate
│   ├── models/
│   ├── utils/            # export_to_hf, mlflow_utils, importing
│   └── compute_numbers/
└── pyproject.toml
```
