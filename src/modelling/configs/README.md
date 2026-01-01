# Configuration Files

Configuration files for training models using the LAMP modelling package.

## Structure

Configs are organized by component:
- `model/` - Model configurations
- `data/` - Data module configurations
- `trainer/` - Training configurations
- `logger/` - Logger configurations

## Usage

Run training with a trainer config:

```bash
python -m modelling.src --config configs/trainer/gru_vae.yaml
```

Trainer configs automatically load matching model/data/logger configs. Override specific components:

```bash
python -m modelling.src \
  --config configs/trainer/gru_vae.yaml \
  --config configs/data/my_custom_data.yaml
```