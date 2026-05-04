# Datamodules

- **`collate.TokenizerCollate`** — batches HF dataset rows, tokenizes a sequence column, and optionally adds `labels=input_ids[:, 1:]` for Trainer compatibility.

Training datasets are built in `modelling.src.training.data.build_datasets` from Hydra `data/*.yaml`.
