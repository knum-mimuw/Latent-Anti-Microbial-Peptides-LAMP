# Configuration Files

Configuration files for training models using the LAMP modelling package.

## Structure

The configs are organized by component:
- `model/` - Model configurations (MetaModule with model, losses, optimizer)
- `data/` - Data module configurations
- `trainer/` - Complete training configurations
- `logger/` - Logger configurations

## Usage

### Using Defaults (Recommended)

Use the modular approach with defaults:

```bash
python -m modelling.src --config configs/trainer/gru_vae.yaml
```

This will automatically load:
- `configs/model/gru_vae.yaml` for model config
- `configs/data/gru_vae.yaml` for data config
- `configs/logger/00.yaml` for logger config

### Using Standalone Config

Use a single complete config file:

```bash
python -m modelling.src --config configs/trainer/gru_vae_full.yaml
```

### Overriding Specific Configs

Override specific components:

```bash
python -m modelling.src \
  --config configs/trainer/gru_vae.yaml \
  --config configs/data/my_custom_data.yaml \
  --config configs/logger/my_logger.yaml
```

## GRU VAE Example

### Model Configuration (`model/gru_vae.yaml`)

Defines:
- GRU VAE model with encoder/decoder
- Loss functions (reconstruction + KL divergence)
- Optimizer (Adam)
- Scheduler (CosineAnnealingLR)

### Data Configuration (`data/gru_vae.yaml`)

Defines:
- Hugging Face dataset loading
- Train/validation splits
- DataLoader settings (batch size, workers, etc.)

### Training Configuration (`trainer/gru_vae.yaml`)

Defines:
- Training hyperparameters (epochs, precision, etc.)
- Uses defaults to import model/data/logger configs

### Complete Configuration (`trainer/gru_vae_full.yaml`)

Standalone file with everything included - useful for:
- Single-file deployments
- Easy sharing
- Quick experiments

## Model Outputs

The GRU VAE model outputs:
- `reconstruction`: Reconstructed logits `[batch_size, seq_len, vocab_size]`
- `mean`: Latent mean `[batch_size, latent_dim]`
- `log_std`: Latent log standard deviation `[batch_size, latent_dim]`

## Loss Configuration

Losses are configured via `loss_manager`:
1. **Reconstruction Loss**: `torch.nn.functional.cross_entropy`
   - Maps `outputs['reconstruction']` → `input` and `batch['input_ids']` → `target`
   - Weight: 1.0

2. **KL Divergence Loss**: `modelling.src.compute_numbers.fns.kl_gaus_unitgauss`
   - Maps `outputs['mean']` → `mean` and `outputs['log_std']` → `log_std`
   - Weight: 0.001 (beta-VAE)

## Customization

### Adjust Hyperparameters

Edit the config files to change:
- Model architecture: `model_kwargs` in model config
- Loss weights: `loss_manager.losses[].weight`
- Optimizer: `optimizer.optimizer_kwargs`
- Scheduler: `scheduler.scheduler_kwargs`
- Training: `trainer` section

### Add Metrics

Create a metrics callback config and add it:

```yaml
# configs/callbacks/metrics.yaml
callbacks:
  - class_path: modelling.src.callbacks.metrics.MetricsCallback
    init_args:
      metric_configs:
        - metric_class_path: torchmetrics.Accuracy
          name: accuracy
          batch_key_mapping:
            input_ids: target
          output_key_mapping:
            reconstruction: input
          stages: ["val"]
```

Then use: `--config configs/callbacks/metrics.yaml`
