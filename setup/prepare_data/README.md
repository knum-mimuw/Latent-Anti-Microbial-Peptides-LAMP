## LAMP data prep utilities

Minimal instructions to create a Hugging Face dataset repository using uv.

### Authentication
Set a Hugging Face token (recommended) or pass `--token` to the command.
```bash
export HF_TOKEN=hf_xxx
```

### Create dataset repo (base command)
```bash
uv run -m setup.prepare_data lamp_short_proteins --private
```

The command prints the ready repo id (e.g., `username/my-dataset`).


