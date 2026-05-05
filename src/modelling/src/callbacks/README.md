# Callbacks

- **`logging.LoggingCallback`** — normalises HF Trainer log keys to `train/` / `eval/` prefixes, filters noisy eval timing metrics, and injects model sub-losses.
- **`iterable_epoch.IterableEpochCallback`** — calls `set_epoch` on iterable/streaming datasets each epoch.
