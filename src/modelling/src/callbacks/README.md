# Callbacks

- **`manifest.ManifestCallback`** — after training, writes `training_manifest.json` and logs checkpoints + manifest to MLflow (requires `report_to` including `mlflow`).
- **`iterable_epoch.IterableEpochCallback`** — calls `set_epoch` on iterable/streaming datasets each epoch.
