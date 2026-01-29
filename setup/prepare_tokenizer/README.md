## LAMP tokenizer preparation utilities

Build and upload protein amino-acid tokenizers to Hugging Face.

### Features

- **1 token = 1 amino acid** — character-level tokenization
- **Fast Rust backend** via HuggingFace Tokenizers
- **AutoTokenizer compatible** — use directly with `transformers`

### Setup

```bash
export HF_TOKEN=hf_xxx
```

### Commands

**Build, upload, and delete in one command:**
```bash
uv run -m setup.prepare_tokenizer run --config setup/prepare_tokenizer/configs/build_and_upload_config.yaml --delete-after
```

**Build tokenizer locally:**
```bash
uv run -m setup.prepare_tokenizer build --output-dir setup/protein-aa-fast-tokenizer
```

**Build using config:**
```bash
uv run -m setup.prepare_tokenizer build --config setup/prepare_tokenizer/configs/build_and_upload_config.yaml
```

**Upload to Hugging Face:**
```bash
uv run -m setup.prepare_tokenizer upload --config setup/prepare_tokenizer/configs/build_and_upload_config.yaml
```

**Upload and delete local directory:**
```bash
uv run -m setup.prepare_tokenizer upload --config setup/prepare_tokenizer/configs/build_and_upload_config.yaml --delete-after
```

### Run Command Flags

The `run` command supports the full workflow with these flags:
- `--skip-build` — Skip building (use existing tokenizer directory)
- `--skip-upload` — Skip uploading to HF Hub
- `--delete-after` — Delete local tokenizer directory after upload

### Vocabulary

The tokenizer uses 28 tokens:
- **Special tokens:** `<PAD>`, `<MASK>`, `<CLS>`, `<SEP>`, `<EOS>`, `<UNK>`
- **Standard amino acids:** A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y
- **Ambiguous codes:** X (any), B (N or D), Z (Q or E)

### Help

```bash
uv run -m setup.prepare_tokenizer --help
uv run -m setup.prepare_tokenizer run --help
uv run -m setup.prepare_tokenizer build --help
uv run -m setup.prepare_tokenizer upload --help
```
