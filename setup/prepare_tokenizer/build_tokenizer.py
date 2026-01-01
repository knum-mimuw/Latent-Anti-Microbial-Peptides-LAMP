# prepare_tokenizer/build_tokenizer.py

from pathlib import Path
from typing import Optional

import typer
from pydantic import BaseModel, Field
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from setup.prepare_data.utils import load_config_file

from .vocab import AA_VOCAB

DEFAULT_OUTPUT_DIR = "setup/protein-aa-fast-tokenizer"


class BuildTokenizerConfig(BaseModel):
    """Configuration for building the tokenizer."""

    output_dir: str = Field(
        DEFAULT_OUTPUT_DIR,
        description="Output directory for the tokenizer files",
    )


def build_tokenizer(
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> PreTrainedTokenizerFast:
    """Build the protein amino-acid tokenizer and save to directory."""
    vocab = {tok: i for i, tok in enumerate(AA_VOCAB)}

    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<UNK>"))

    # Amino-acid level split (1 char = 1 token)
    tokenizer.pre_tokenizer = Split(pattern="", behavior="isolated")

    # <CLS> SEQ <EOS>
    tokenizer.post_processor = TemplateProcessing(
        single="<CLS> $A <EOS>",
        pair="<CLS> $A <SEP> $B <EOS>",
        special_tokens=[
            ("<CLS>", vocab["<CLS>"]),
            ("<SEP>", vocab["<SEP>"]),
            ("<EOS>", vocab["<EOS>"]),
        ],
    )

    # Wrap as HuggingFace fast tokenizer
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<PAD>",
        mask_token="<MASK>",
        cls_token="<CLS>",
        sep_token="<SEP>",
        eos_token="<EOS>",
        unk_token="<UNK>",
        model_input_names=["input_ids", "attention_mask"],
    )

    hf_tokenizer.save_pretrained(output_dir)
    typer.echo(f"✓ Tokenizer saved to {output_dir}/")

    return hf_tokenizer


def build_tokenizer_command(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to YAML config file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    output_dir: str = typer.Option(
        DEFAULT_OUTPUT_DIR,
        "--output-dir",
        "-o",
        help="Output directory for tokenizer files",
    ),
) -> None:
    """Build the protein amino-acid fast tokenizer."""
    if config:
        raw = load_config_file(config)
        build_cfg = BuildTokenizerConfig(**raw.get("build", {}))
        output_dir = build_cfg.output_dir

    tokenizer = build_tokenizer(output_dir)

    # Sanity check
    out = tokenizer(
        ["MKTLLILAVAVCSAA", "ACDEFGHIK"],
        padding=True,
        return_tensors="pt",
    )
    typer.echo("Sanity check:")
    typer.echo(str(out))


if __name__ == "__main__":
    build_tokenizer_command()
