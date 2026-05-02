"""HydrAMP amino-acid tokenizer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from transformers import PreTrainedTokenizer

DEFAULT_TOKENS = ["<pad>"] + list("ACDEFGHIKLMNPQRSTVWY")


class HydrAMPAATokenizer(PreTrainedTokenizer):
    """Character-level amino-acid tokenizer for HydrAMP."""

    vocab_files_names = {"vocab_file": "vocab.json"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, vocab_file: str | None = None, **kwargs: Any) -> None:
        if vocab_file is not None:
            payload = json.loads(Path(vocab_file).read_text())
            self._token_to_id = {str(k): int(v) for k, v in payload.items()}
        else:
            self._token_to_id = {token: idx for idx, token in enumerate(DEFAULT_TOKENS)}
        self._id_to_token = {idx: token for token, idx in self._token_to_id.items()}
        self._strict_unknown = bool(kwargs.pop("strict_unknown", True))

        kwargs.setdefault("pad_token", "<pad>")
        kwargs.setdefault("model_max_length", 25)
        kwargs.setdefault("padding_side", "right")
        super().__init__(**kwargs)

        if self.pad_token not in self._token_to_id:
            raise ValueError(f"pad token '{self.pad_token}' not present in tokenizer vocabulary.")
        self.pad_token_id = self._token_to_id[self.pad_token]

    @property
    def vocab_size(self) -> int:
        return len(self._token_to_id)

    def get_vocab(self) -> dict[str, int]:
        return dict(self._token_to_id)

    def _tokenize(self, text: str) -> list[str]:
        return list(text.strip().upper())

    def _convert_token_to_id(self, token: str) -> int:
        if token in self._token_to_id:
            return self._token_to_id[token]
        if self._strict_unknown:
            raise ValueError(f"Unknown amino-acid token '{token}'.")
        return self.pad_token_id

    def _convert_id_to_token(self, index: int) -> str:
        return self._id_to_token.get(index, self.pad_token)

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return "".join(token for token in tokens if token != self.pad_token)

    def build_inputs_with_special_tokens(self, token_ids_0: list[int], token_ids_1: list[int] | None = None) -> list[int]:
        if token_ids_1 is not None:
            return token_ids_0 + token_ids_1
        return token_ids_0

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None) -> tuple[str]:
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        filename = "vocab.json" if filename_prefix is None else f"{filename_prefix}-vocab.json"
        vocab_path = save_dir / filename
        vocab_path.write_text(json.dumps(self._token_to_id, indent=2, sort_keys=True) + "\n")
        return (str(vocab_path),)

