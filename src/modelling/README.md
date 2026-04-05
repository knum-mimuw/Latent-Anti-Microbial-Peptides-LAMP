# Modelling Package

PyTorch Lightning-based framework for training models with configurable losses and metrics.

## Running Training

Use PyTorch Lightning CLI with configuration files:

```bash
uv run modelling fit --config configs/trainer/grugru_vae.yaml
```

Or use the module directly:

```bash
uv run python -m modelling.src fit --config configs/trainer/grugru_vae.yaml
```

Other Lightning commands:

```bash
uv run modelling validate --config configs/trainer/grugru_vae.yaml
uv run modelling test --config configs/trainer/grugru_vae.yaml
uv run modelling predict --config configs/trainer/grugru_vae.yaml
```

## Experiment Tracking with MLflow

By default, the logger configs use MLflow. Set `MLFLOW_TRACKING_URI` in your
environment (via `.env` / direnv) to control where runs are stored:

```bash
# File-based (no server needed, default if unset)
export MLFLOW_TRACKING_URI=file:./mlflow-store

# Or point at a local MLflow server
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

Use the experiment-specific logger:

```bash
uv run modelling fit --config configs/trainer/grugru_vae.yaml
# auto-loads configs/logger/grugru_vae.yaml (experiment_name: grugru_vae)
```

Or the generic logger override used by the ZenML pipelines:

```bash
uv run modelling fit \
  --config configs/trainer/grugru_vae.yaml \
  --config configs/logger/mlflow_local.yaml
```

### Checkpoints

Add the explicit checkpoint callback for fine-grained control:

```bash
uv run modelling fit \
  --config configs/trainer/grugru_vae.yaml \
  --config configs/callbacks/checkpoint.yaml
```

When the training-manifest callback is active, this saves the top checkpoints,
writes a deterministic `training_manifest.json`, and logs both the checkpoint
directory and manifest into the active MLflow run.

### MLflow Utilities

`modelling.src.utils.mlflow_utils` provides standalone helpers (no ZenML needed):

- `get_mlflow_client()` -- configured from `MLFLOW_TRACKING_URI`
- `download_checkpoint(run_id, artifact_path)` -- pull a checkpoint from an MLflow run
- `download_artifact(run_id, artifact_path)` -- pull any artifact from an MLflow run
- `download_config(run_id)` -- pull the logged config YAML
- `log_checkpoint_artifact(run_id, checkpoint_path)` -- push a `.ckpt` to a run
- `log_artifact_directory(run_id, local_dir, artifact_path)` -- push a whole artifact subtree
- `list_experiments()` / `list_runs(experiment_name)` -- browse runs

## Configuration Structure

Configs are organized by component in `configs/`:

### Trainer Config (`trainer/*.yaml`)
- Training hyperparameters (epochs, precision, devices, etc.)
- Lightning CLI automatically loads matching configs from other directories

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

## Config Composition

Lightning CLI automatically composes configs based on naming conventions. When you specify a trainer config:

```bash
modelling fit --config configs/trainer/grugru_vae.yaml
```

It automatically loads matching configs:
- `configs/model/grugru_vae.yaml` -- model configuration
- `configs/data/grugru_vae.yaml` -- data configuration
- `configs/logger/grugru_vae.yaml` -- logger configuration

### Overriding Configs

Override specific components by passing multiple config files (later configs override earlier ones):

```bash
modelling fit \
  --config configs/trainer/grugru_vae.yaml \
  --config configs/data/my_custom_data.yaml
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
