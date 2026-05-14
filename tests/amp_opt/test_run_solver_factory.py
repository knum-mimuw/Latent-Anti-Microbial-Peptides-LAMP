"""``run.import_from_path`` and solver factories resolve to ``StepByStepSolver``."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.black_box_information import BlackBoxInformation

from amp_opt.run import import_from_path
from amp_opt.solver_factories import protein_random_mutation
from amp_opt.step_by_step_solver import StepByStepSolver


class _TinyBlackBox(AbstractBlackBox):
    def __init__(self) -> None:
        super().__init__()
        from amp_opt.sequence_constants import STANDARD_AA_ORDER

        self._info = BlackBoxInformation(
            "tiny",
            3,
            True,
            True,
            True,
            list(STANDARD_AA_ORDER),
        )

    def get_black_box_info(self) -> BlackBoxInformation:
        return self._info

    def _black_box(self, x: np.ndarray, context: object | None = None) -> np.ndarray:
        return np.zeros((x.shape[0], 1), dtype=np.float64)


def test_import_from_path_factory_string() -> None:
    f = import_from_path("amp_opt.solver_factories:protein_random_mutation")
    assert f is protein_random_mutation


def test_protein_random_mutation_returns_step_solver() -> None:
    bb = _TinyBlackBox()
    x0 = np.array([list("ACD")], dtype=object)
    y0 = np.array([[0.0]])
    sol = protein_random_mutation(
        bb,
        x0,
        y0,
        {"n_mutations": 1, "top_k": 1, "greedy": True, "batch_size": 1},
    )
    assert isinstance(sol, StepByStepSolver)


def test_import_from_path_rejects_bad_string() -> None:
    with pytest.raises(ValueError, match="module:callable"):
        import_from_path("bad")


@pytest.fixture
def fake_hf_model() -> object:
    class _Enc:
        def encode(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            b = input_ids.shape[0]
            mean = torch.zeros(b, 4, device=input_ids.device, dtype=torch.float32)
            log_std = torch.zeros_like(mean)
            return mean, log_std

    class _Cfg:
        sequence_length = 2
        vocab_size = 3
        condition_dim = 2

    class _Model:
        def __init__(self) -> None:
            self.config = _Cfg()
            self.encoder = _Enc()

        def to(self, _device: torch.device) -> object:
            return self

        def eval(self) -> object:
            return self

        def forward_latent_positions(
            self,
            z: torch.Tensor,
            condition: torch.Tensor | None = None,
            *,
            return_logits: bool = True,
        ) -> object:
            del condition
            logits = torch.zeros(
                z.shape[0],
                self.config.sequence_length,
                self.config.vocab_size,
                device=z.device,
                dtype=z.dtype,
            )
            if return_logits:
                return type("O", (), {"logits": logits})()
            raise AssertionError

    return _Model()


@pytest.fixture
def fake_tokenizer() -> object:
    class _Tok:
        pad_token = "<pad>"
        pad_token_id = 0

        def convert_id_to_token(self, index: int) -> str:
            return {0: "<pad>", 1: "A", 2: "C"}[int(index)]

        def __call__(
            self,
            seq: str,
            *,
            add_special_tokens: bool = False,
            return_tensors: str = "pt",
            **_kwargs: object,
        ) -> dict[str, torch.Tensor]:
            del add_special_tokens, _kwargs
            if return_tensors != "pt":
                raise ValueError(return_tensors)
            mp = {"A": 1, "C": 2}
            ids = [mp[c] for c in seq.upper()[:2]]
            while len(ids) < 2:
                ids.append(1)
            return {"input_ids": torch.tensor([ids[:2]], dtype=torch.long)}

    return _Tok()


def test_build_mutang_hydramp_solver_returns_mutang(
    fake_hf_model: object,
    fake_tokenizer: object,
) -> None:
    from unittest.mock import patch

    from amp_opt.mutang_hydramp_init import build_mutang_hydramp_solver
    from amp_opt.mutang_solver import ProteinMutangUniformMutation

    alphabet = ["<pad>", "A", "C", "D"]  # length >= vocab_size; used only for mock consistency

    class _BB(AbstractBlackBox):
        def __init__(self) -> None:
            super().__init__()
            self._info = BlackBoxInformation("m", 2, True, True, True, alphabet)

        def get_black_box_info(self) -> BlackBoxInformation:
            return self._info

        def _black_box(self, x: np.ndarray, context: object | None = None) -> np.ndarray:
            return np.zeros((x.shape[0], 1), dtype=np.float64)

    bb = _BB()
    x0 = np.array([["A", "C"]], dtype=object)
    y0 = np.array([[0.0]])
    kwargs = {
        "hf_model_repo_id": "org/model",
        "hf_tokenizer_repo_id": "org/tok",
        "device": "cpu",
        "min_number_of_directions": 1,
        "token_threshold": 0.0,
    }
    with (
        patch("transformers.AutoModel.from_pretrained", return_value=fake_hf_model),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=fake_tokenizer),
    ):
        sol = build_mutang_hydramp_solver(bb, x0, y0, kwargs)
    assert isinstance(sol, ProteinMutangUniformMutation)
