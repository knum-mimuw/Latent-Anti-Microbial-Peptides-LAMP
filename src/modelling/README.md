# Modelling Package

PyTorch Lightning-based framework for training models with configurable losses and metrics.

## Running Training

Run commands from the **repository root** (the directory that contains `src/modelling/`).
You must pass **every** component the CLI needs: at minimum **trainer**, **model**, and **data**.
Nothing is merged or discovered automatically from filenames.

```bash
uv run modelling fit \
  --config src/modelling/configs/trainer/grugru_vae.yaml \
  --config src/modelling/configs/model/grugru_vae.yaml \
  --config src/modelling/configs/data/grugru_vae.yaml \
  --config src/modelling/configs/logger/grugru_vae.yaml \
  --config src/modelling/configs/callbacks/checkpoint.yaml
```

Or the module entry point:

```bash
uv run python -m modelling.src fit \
  --config src/modelling/configs/trainer/grugru_vae.yaml \
  --config src/modelling/configs/model/grugru_vae.yaml \
  --config src/modelling/configs/data/grugru_vae.yaml \
  --config src/modelling/configs/logger/grugru_vae.yaml \
  --config src/modelling/configs/callbacks/checkpoint.yaml
```

Other Lightning subcommands need the same `--config` stack (trainer + model + data,
plus logger/callbacks as needed):

```bash
uv run modelling validate \
  --config src/modelling/configs/trainer/grugru_vae.yaml \
  --config src/modelling/configs/model/grugru_vae.yaml \
  --config src/modelling/configs/data/grugru_vae.yaml
```

### Shorter paths (optional)

If you `cd src/modelling`, you can use `configs/...` instead of `src/modelling/configs/...`.

## Experiment Tracking with MLflow

Set `MLFLOW_TRACKING_URI` in your environment (via `.env` / direnv) to control where
runs are stored:

```bash
# File-based (no server needed, default if unset)
export MLFLOW_TRACKING_URI=sqlite:///mlflow-store/mlflow.db

# Or point at a local MLflow server
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

Use `src/modelling/configs/logger/grugru_vae.yaml` for a fixed experiment name, or
`src/modelling/configs/logger/mlflow_local.yaml` when something else (e.g. ZenML) overrides the
experiment name at run time:

```bash
uv run modelling fit \
  --config src/modelling/configs/trainer/grugru_vae.yaml \
  --config src/modelling/configs/model/grugru_vae.yaml \
  --config src/modelling/configs/data/grugru_vae.yaml \
  --config src/modelling/configs/logger/mlflow_local.yaml \
  --config src/modelling/configs/callbacks/checkpoint.yaml
```

When the training-manifest callback is active, training saves top checkpoints,
writes `training_manifest.json`, and logs the checkpoint directory and manifest
into the active MLflow run.

### MLflow Utilities

`modelling.src.utils.mlflow_utils` provides standalone helpers (no ZenML needed):

- `get_mlflow_client()` -- configured from `MLFLOW_TRACKING_URI`
- `download_artifact(run_id, artifact_path)` -- pull any artifact (checkpoint, config, etc.) from an MLflow run
- `download_config(run_id)` -- pull the logged config YAML
- `log_checkpoint_artifact(run_id, checkpoint_path)` -- push a `.ckpt` to a run
- `log_artifact_directory(run_id, local_dir, artifact_path)` -- push a whole artifact subtree
- `list_experiments()` / `list_runs(experiment_name)` -- browse runs

## Configuration Structure

Configs live under `src/modelling/configs/`:

### Trainer Config (`trainer/*.yaml`)

- Training hyperparameters (epochs, precision, devices, etc.)

### Model Config (`model/*.yaml`)

- Model class path and initialization arguments
- Loss manager with multiple losses (each with class path, weight, key mappings)
- Optimizer configuration (class path and kwargs)
- Scheduler configuration (optional)

### Data Config (`data/*.yaml`)

- Dataset loading configuration
- DataLoader settings (batch size, workers, etc.)
- Train/validation splits

### Logger Config (`logger/*.yaml`)

- `grugru_vae.yaml` -- MLflow logger with experiment name `grugru_vae`
- `mlflow_local.yaml` -- generic MLflow logger used by the pipelines, with run-scoped overrides applied at execution time

### Callbacks Config (`callbacks/*.yaml`)

- `checkpoint.yaml` -- ModelCheckpoint with `val/loss` monitoring

## Merging configs

Pass multiple `--config` files; **later files override earlier ones** for overlapping
keys. There is no implicit loading of `model/` or `data/` YAMLs based on the
trainer filename.

```bash
uv run modelling fit \
  --config src/modelling/configs/trainer/grugru_vae.yaml \
  --config src/modelling/configs/model/grugru_vae.yaml \
  --config src/modelling/configs/data/grugru_vae.yaml \
  --config src/modelling/configs/data/my_custom_data.yaml
```

### Config Structure

Each config file follows Lightning CLI format with top-level keys:

- `trainer:` - Trainer hyperparameters
- `model:` - Model configuration (class_path, init_args)
- `data:` - Data module configuration (class_path, init_args)
- `logger:` - Logger configuration (list of loggers with class_path, init_args)

## Package Structure

```
src/modelling/
├── src/
│   ├── __main__.py          # Lightning CLI entry point
│   ├── metamodule/          # Core training module
│   │   ├── metamodule.py    # MetaModule and StepOutput
│   │   ├── loss_manager.py  # Loss computation and management
│   │   └── utils/          # Utilities (argument mapping, lightning config)
│   ├── callbacks/           # PyTorch Lightning callbacks
│   │   ├── metrics.py      # MetricsCallback
│   │   └── training_manifest.py # writes deterministic MLflow manifest/artifacts
│   ├── datamodules/         # Data loading modules
│   │   └── seq_dm.py       # SequenceDataModule
│   ├── models/              # Model implementations
│   │   └── aes/            # Autoencoder models
│   ├── utils/               # Shared utilities
│   │   ├── importing.py    # Dynamic imports
│   │   ├── mlflow_utils.py # MLflow checkpoint/artifact helpers
│   │   └── export_to_hf.py # HuggingFace Hub export
│   └── compute_numbers/     # Computation functions
└── configs/                 # Configuration files
    ├── trainer/            # Training configurations
    ├── model/              # Model configurations
    ├── data/               # Data configurations
    ├── logger/             # Logger configurations (MLflow)
    └── callbacks/          # Callback configurations (checkpoints)
```
