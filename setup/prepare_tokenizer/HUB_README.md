---
language:
  - en
tags:
  - protein
  - amino-acid
  - tokenizer
  - biology
license: mit
library_name: transformers
---

# Protein Amino-Acid Fast Tokenizer

Fast Rust-backed tokenizer for protein sequences.

## Features

- **1 token = 1 amino acid** — character-level tokenization
- **Fast Rust backend** — efficient processing via HuggingFace Tokenizers
- **Transformer-ready** — compatible with `AutoTokenizer`

## Usage

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("pszmk/protein-aa-fast-tokenizer")

# Single sequence
tokens = tokenizer("MKTLLILAVAVCSAA")
print(tokens)
# {'input_ids': [2, 16, 14, ...], 'attention_mask': [1, 1, ...]}

# Batch with padding
batch = tokenizer(
    ["MKTLLILAVAVCSAA", "ACDEFGHIK"],
    padding=True,
    return_tensors="pt",
)
```

## Vocabulary

| ID | Token | Description |
|----|-------|-------------|
| 0 | `<PAD>` | Padding |
| 1 | `<MASK>` | Masked token |
| 2 | `<CLS>` | Classification / Start |
| 3 | `<SEP>` | Separator |
| 4 | `<EOS>` | End of sequence |
| 5 | `<UNK>` | Unknown |
| 6-25 | A-Y | Standard amino acids |
| 26 | X | Any amino acid |
| 27 | B | Asparagine or Aspartic acid |
| 28 | Z | Glutamine or Glutamic acid |

## Template Processing

- **Single sequence:** `<CLS> SEQUENCE <EOS>`
- **Pair sequences:** `<CLS> SEQ_A <SEP> SEQ_B <EOS>`

## Citation

Part of the LAMP (Latent Anti-Microbial Peptides) project.


