# Modelling Package

PyTorch Lightning-based framework for training models with configurable losses and metrics.

## Running Training

Use PyTorch Lightning CLI with configuration files:

```bash
modelling fit --config configs/trainer/grugru_vae.yaml
```

Or use the module directly:

```bash
python -m modelling.src fit --config configs/trainer/grugru_vae.yaml
```

Other Lightning commands:

```bash
modelling validate --config configs/trainer/grugru_vae.yaml
modelling test --config configs/trainer/grugru_vae.yaml
modelling predict --config configs/trainer/grugru_vae.yaml
```

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
- Logger configuration (W&B, TensorBoard, etc.)

## Config Composition

Lightning CLI automatically composes configs based on naming conventions. When you specify a trainer config:

```bash
modelling fit --config configs/trainer/grugru_vae.yaml
```

It automatically loads matching configs:
- `configs/model/grugru_vae.yaml` → model configuration
- `configs/data/grugru_vae.yaml` → data configuration
- `configs/logger/grugru_vae.yaml` → logger configuration

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
│   │   └── metrics.py      # MetricsCallback
│   ├── datamodules/         # Data loading modules
│   │   └── seq_dm.py       # SequenceDataModule
│   ├── models/              # Model implementations
│   │   └── aes/            # Autoencoder models
│   ├── utils/               # Shared utilities
│   │   └── importing.py    # Dynamic imports and Hugging Face loading
│   └── compute_numbers/     # Computation functions
└── configs/                 # Configuration files
    ├── trainer/            # Training configurations
    ├── model/              # Model configurations
    ├── data/               # Data configurations
    └── logger/             # Logger configurations
```

