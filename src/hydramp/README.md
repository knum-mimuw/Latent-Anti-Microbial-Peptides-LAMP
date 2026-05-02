# hydramp

**hydramp** is the standalone Python package for **HydrAMP** in the LAMP (latent
antimicrobial peptide modelling) workspace: encoder/decoder weights, Hugging Face export
(`AutoModel` / `AutoTokenizer` with `trust_remote_code=True`), and parity checks. Hub model
repos get a generated `README.md` with the real `repo_id`, LAMP context, and copy-paste
encode/decode examples whenever you upload via `export_to_hf` or `publish_from_config`.

## Acknowledging HydrAMP

If you use this architecture or compare to HydrAMP, cite the original **Nature Communications**
paper (Szymczak *et al.*, 2023). The **Citation** section below is the same text appended to
Hugging Face **model** and **tokenizer** repo `README.md` files when you upload with
`export_to_hf`, `export_tokenizer_to_hf`, or `publish_from_config` (see
[`hydramp/citation.py`](src/hydramp/src/hydramp/citation.py)).

## Layout

- `src/hydramp/`: Python package modules
- `src/hydramp/configs/`: YAML configs for Hub uploads (default branch: `main`)
- `weights/`: local model weights (gitignored)

## Citation

The **HydrAMP** architecture and original model were introduced by Szymczak *et al.* in *Nature Communications* (2023). When you refer to HydrAMP or build on this work, please cite:

```bibtex
@article{szymczak_discovering_2023,
  title = {Discovering highly potent antimicrobial peptides with deep generative model {HydrAMP}},
  volume = {14},
  issn = {2041-1723},
  url = {https://www.nature.com/articles/s41467-023-36994-z},
  doi = {10.1038/s41467-023-36994-z},
  abstract = {Antimicrobial peptides emerge as compounds that can alleviate the global health hazard of antimicrobial resistance, prompting a need for novel computational approaches to peptide generation. Here, we propose HydrAMP, a conditional variational autoencoder that learns lower-dimensional, continuous representation of peptides and captures their antimicrobial properties. The model disentangles the learnt representation of a peptide from its antimicrobial conditions and leverages parameter-controlled creativity. HydrAMP is the first model that is directly optimized for diverse tasks, including unconstrained and analogue generation and outperforms other approaches in these tasks. An additional preselection procedure based on ranking of generated peptides and molecular dynamics simulations increases experimental validation rate. Wet-lab experiments on five bacterial strains confirm high activity of nine peptides generated as analogues of clinically relevant prototypes, as well as six analogues of an inactive peptide. HydrAMP enables generation of diverse and potent peptides, making a step towards resolving the antimicrobial resistance crisis.},
  language = {en},
  number = {1},
  journal = {Nature Communications},
  author = {Szymczak, Paulina and Możejko, Marcin and Grzegorzek, Tomasz and Jurczak, Radosław and Bauer, Marta and Neubauer, Damian and Sikora, Karol and Michalski, Michał and Sroka, Jacek and Setny, Piotr and Kamysz, Wojciech and Szczurek, Ewa},
  month = mar,
  year = {2023},
  keywords = {Computational models, Machine learning, Protein design},
  pages = {1453},
}
```

- **DOI:** [10.1038/s41467-023-36994-z](https://doi.org/10.1038/s41467-023-36994-z)
- **Article:** [nature.com](https://www.nature.com/articles/s41467-023-36994-z)

Canonical BibTeX and Markdown fragment: [`hydramp/citation.py`](src/hydramp/src/hydramp/citation.py) (`HYDRAMP_ORIGINAL_ARTICLE_BIBTEX`, `HYDRAMP_README_CITATION_SECTION`).

## Publish from YAML (`main`)

Edit `src/hydramp/configs/hydramp_model_hub.yaml` and
`src/hydramp/configs/hydramp_tokenizer_hub.yaml` (`repo_id`, optional `tag`,
`private`). Both default to **`revision: main`** so uploads land on each repo’s
default branch without creating a release branch.

```bash
# from repository root
uv run --package hydramp python -m hydramp.publish_from_config \
  src/hydramp/configs/hydramp_model_hub.yaml

uv run --package hydramp python -m hydramp.publish_from_config \
  src/hydramp/configs/hydramp_tokenizer_hub.yaml
```

Model config `weights_dir` is relative to the YAML file unless you use an
absolute path (default `../weights` → `src/hydramp/weights`).

## Export to Hugging Face

HydrAMP can be exported as a Hugging Face remote-code model that loads with
`AutoModel.from_pretrained(..., trust_remote_code=True)`.

Expected local weights in `<weights-dir>`:

- `encoder_weights.pickle`
- `decoder_weights.pickle`

Note: legacy split-weight loading is handled by the export script as a
migration utility; `HydrAMPModel` itself is HF-native and no longer exposes
local pickle-loading methods.

Latents: `model.encoder.encode(input_ids)` (same as `model(...)`’s `mean` /
`log_std` path). Decode from `z` in the same style as GRUVAE:
`model.forward_latent_positions(z, return_logits=True).logits`, or
`model.decode_to_token_ids(z)` for greedy IDs (`num_steps` must match
`config.sequence_length` when passed explicitly).

Set `HF_TOKEN` (see repository root `.env-default`) for Hub uploads, or use
`huggingface-cli login`.

### Upload model (remote-code bundle)

```bash
uv run --package hydramp python -m hydramp.export_to_hf \
  --weights-dir src/hydramp/weights \
  --repo-id your-org/hydramp \
  --revision release/run-20260501 \
  --tag run-20260501
```

### Upload HydrAMP AA tokenizer

```bash
uv run --package hydramp python -m hydramp.export_tokenizer_to_hf \
  --repo-id your-org/hydramp-aa-tokenizer \
  --revision release/run-20260501 \
  --tag run-20260501
```

### Write export bundles locally (no Hub)

Use the same trees as an upload, for parity checks or manual `huggingface-cli upload`:

```bash
uv run --package hydramp python -m hydramp.export_to_hf \
  --weights-dir src/hydramp/weights \
  --local-export-dir ./build/hydramp-export

uv run --package hydramp python -m hydramp.export_tokenizer_to_hf \
  --local-export-dir ./build/hydramp-aa-tokenizer-export
```

### Deterministic parity (CPU, `eval()`)

Compares split `hydramp.HydrAMPEncoder` / `HydrAMPDecoder` with loaded pickles,
`HydrAMPModel` with `_load_legacy_weights`, and `AutoModel.from_pretrained` on a
temporary local export (same code path as Hub). Run with `model.eval()` on CPU
and fixed seeds inside the script.

```bash
uv run --package hydramp python -m hydramp.verify_hf_parity \
  --weights-dir src/hydramp/weights \
  --atol 1e-5
```

Pass criteria: max absolute error ≤ `--atol` for encoder `(mean, log_std)` and
decoder logits; greedy argmax token IDs must match.

### Load from Hugging Face

```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "your-org/hydramp",
    revision="run-20260501",
    trust_remote_code=True,
)
```

### Load tokenizer

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "your-org/hydramp-aa-tokenizer",
    revision="run-20260501",
    trust_remote_code=True,
)
```

## Export HydrAMP AA tokenizer

See **Upload HydrAMP AA tokenizer** and **Write export bundles locally** above.

## Notebook

`notebooks/grugru_vae_hf_demo.ipynb` loads HydrAMP from the Hub by default. For
private repos, authenticate first. To use local export directories, set
`HYDRAMP_LOCAL_MODEL_DIR` and `HYDRAMP_LOCAL_TOKENIZER_DIR` (documented in root
`.env-default`).
