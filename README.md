# Setup

## Environment with uv

1. Install uv (if not installed):
   - See docs: `https://docs.astral.sh/uv/getting-started/installation/`

2. Create a virtual environment in the project:
   ```bash
    uv venv
   ```

3. Sync dependencies from `pyproject.toml`:
   ```bash
   uv sync
   ```

4. Run the package with the PyTorch Lightning CLI:
   ```bash
    uv run lamp --help
   ```

### CLI usage

- Show help:
  ```bash
  uv run lamp --help
  ```
- Show version (same as project version):
  ```bash
  uv run lamp --version
  ```
- Typical training invocation (examples; adjust to your model/data classes):
  ```bash
  # Use subclass discovery (enabled in __main__.py) and pass config via CLI
  uv run lamp fit --trainer.max_epochs=5 --trainer.accelerator=cpu
  
  # Or provide a YAML config
  uv run lamp fit --config config.yaml
  ```

## Linting & formatting (Ruff)

- Install dev tools:
  ```bash
  uv sync --dev
  ```
- Lint the code:
  ```bash
  uv run ruff check .
  ```
- Format the code:
  ```bash
  uv run ruff format .
  ```

## Project layout

```
/home/prz/Latent-Anti-Microbial-Peptides-LAMP
├── pyproject.toml
├── README.md
└── src/
    └── lamp/
        ├── __init__.py
        └── __main__.py
```

