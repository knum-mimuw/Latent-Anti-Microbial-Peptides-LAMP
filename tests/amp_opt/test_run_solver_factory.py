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

        def convert_ids_to_tokens(self, index: int) -> str:
            return {0: "<pad>", 1: "A", 2: "C"}[int(index)]

        def __call__(
            self,
            seq: str,
            *,
            add_special_tokens: bool = False,
            return_tensors: str = "pt",
            **_kwargs: object,
        ) -> dict[str, torch.Tensor]:
            del add_special_tokens
            if return_tensors != "pt":
                raise ValueError(return_tensors)
            mp = {"A": 1, "C": 2}
            max_len = int(_kwargs.get("max_length", 2))
            ids = [mp[c] for c in seq.upper() if c in mp]
            while len(ids) < max_len:
                ids.append(self.pad_token_id)
            return {"input_ids": torch.tensor([ids[:max_len]], dtype=torch.long)}

    return _Tok()


def _make_bb(
    alphabet: list[str], seq_len: int
) -> AbstractBlackBox:
    class _BB(AbstractBlackBox):
        def __init__(self) -> None:
            super().__init__()
            self._info = BlackBoxInformation("m", seq_len, True, True, True, alphabet)

        def get_black_box_info(self) -> BlackBoxInformation:
            return self._info

        def _black_box(self, x: np.ndarray, context: object | None = None) -> np.ndarray:
            return np.zeros((x.shape[0], 1), dtype=np.float64)

    return _BB()


def _base_mutang_kwargs() -> dict:
    return {
        "hf_model_repo_id": "org/model",
        "hf_tokenizer_repo_id": "org/tok",
        "device": "cpu",
        "min_number_of_directions": 1,
        "token_threshold": 0.0,
    }


@pytest.fixture
def fake_hf_model_with_decode() -> object:
    """Fake HydrAMP model that also supports decode_to_token_ids (for latent sampling)."""

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
            logits[:, 0, 1] = 1.0  # prefer token 1 ("A") at pos 0
            logits[:, 1, 2] = 1.0  # prefer token 2 ("C") at pos 1
            if return_logits:
                return type("O", (), {"logits": logits})()
            raise AssertionError

        def decode_to_token_ids(self, z: torch.Tensor) -> torch.Tensor:
            out = self.forward_latent_positions(z, return_logits=True)
            return out.logits.argmax(dim=-1)

    return _Model()


def test_build_latent_sampling_hydramp_solver_returns_solver(
    fake_hf_model_with_decode: object,
    fake_tokenizer: object,
) -> None:
    from unittest.mock import patch

    from amp_opt.latent_sampling_hydramp_init import build_latent_sampling_hydramp_solver
    from amp_opt.latent_sampling_solver import ProteinLatentSampling

    alphabet = ["<pad>", "A", "C", "D"]
    bb = _make_bb(alphabet, 2)
    x0 = np.array([["A", "C"]], dtype=object)
    y0 = np.array([[0.0]])
    with (
        patch("transformers.AutoModel.from_pretrained", return_value=fake_hf_model_with_decode),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=fake_tokenizer),
    ):
        sol = build_latent_sampling_hydramp_solver(
            bb, x0, y0,
            {
                "hf_model_repo_id": "org/model",
                "hf_tokenizer_repo_id": "org/tok",
                "device": "cpu",
                "tau": 2.0,
            },
        )
    assert isinstance(sol, ProteinLatentSampling)
    assert sol.tau == 2.0


def test_latent_sampling_solver_step_produces_valid_sequence(
    fake_hf_model_with_decode: object,
    fake_tokenizer: object,
) -> None:
    """Run 3 solver steps and verify outputs are valid AA sequences."""
    from unittest.mock import patch

    from amp_opt.latent_sampling_hydramp_init import build_latent_sampling_hydramp_solver

    alphabet = ["<pad>", "A", "C", "D"]
    bb = _make_bb(alphabet, 2)
    x0 = np.array([["A", "C"]], dtype=object)
    y0 = np.array([[0.0]])
    with (
        patch("transformers.AutoModel.from_pretrained", return_value=fake_hf_model_with_decode),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=fake_tokenizer),
    ):
        sol = build_latent_sampling_hydramp_solver(
            bb, x0, y0,
            {
                "hf_model_repo_id": "org/model",
                "hf_tokenizer_repo_id": "org/tok",
                "device": "cpu",
                "tau": 1.0,
            },
        )

    valid_tokens = {"A", "C"}
    for _ in range(3):
        x, y = sol.step()
        assert x.shape == (1, 2)
        assert y.shape == (1, 1)
        for tok in x.reshape(-1):
            assert str(tok) in valid_tokens, f"Unexpected token {tok!r}"


def test_latent_sampling_solver_tau0_reconstructs(
    fake_hf_model_with_decode: object,
    fake_tokenizer: object,
) -> None:
    """With tau=0 (no noise), every sample should decode identically."""
    from unittest.mock import patch

    from amp_opt.latent_sampling_hydramp_init import build_latent_sampling_hydramp_solver

    alphabet = ["<pad>", "A", "C", "D"]
    bb = _make_bb(alphabet, 2)
    x0 = np.array([["A", "C"]], dtype=object)
    y0 = np.array([[0.0]])
    with (
        patch("transformers.AutoModel.from_pretrained", return_value=fake_hf_model_with_decode),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=fake_tokenizer),
    ):
        sol = build_latent_sampling_hydramp_solver(
            bb, x0, y0,
            {
                "hf_model_repo_id": "org/model",
                "hf_tokenizer_repo_id": "org/tok",
                "device": "cpu",
                "tau": 0.0,
            },
        )

    results = set()
    for _ in range(5):
        x, _y = sol.step()
        results.add(tuple(x.reshape(-1)))
    assert len(results) == 1, f"tau=0 should produce identical outputs, got {results}"


def test_build_mutang_hydramp_solver_returns_mutang(
    fake_hf_model: object,
    fake_tokenizer: object,
) -> None:
    from unittest.mock import patch

    from amp_opt.mutang_hydramp_init import build_mutang_hydramp_solver
    from amp_opt.mutang_solver import ProteinMutangUniformMutation

    alphabet = ["<pad>", "A", "C", "D"]
    bb = _make_bb(alphabet, 2)
    x0 = np.array([["A", "C"]], dtype=object)
    y0 = np.array([[0.0]])
    with (
        patch("transformers.AutoModel.from_pretrained", return_value=fake_hf_model),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=fake_tokenizer),
    ):
        sol = build_mutang_hydramp_solver(bb, x0, y0, _base_mutang_kwargs())
    assert isinstance(sol, ProteinMutangUniformMutation)
    assert sol._mutable_prefix is None


def test_build_mutang_hydramp_solver_pads_short_seed(
    fake_hf_model: object,
    fake_tokenizer: object,
) -> None:
    """x0 length 1 < model sequence_length 2 triggers padding."""
    from unittest.mock import patch

    from amp_opt.mutang_hydramp_init import build_mutang_hydramp_solver
    from amp_opt.mutang_solver import ProteinMutangUniformMutation

    alphabet = ["<pad>", "A", "C", "D"]
    bb = _make_bb(alphabet, 1)
    x0 = np.array([["A"]], dtype=object)
    y0 = np.array([[0.0]])
    with (
        patch("transformers.AutoModel.from_pretrained", return_value=fake_hf_model),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=fake_tokenizer),
    ):
        sol = build_mutang_hydramp_solver(bb, x0, y0, _base_mutang_kwargs())
    assert isinstance(sol, ProteinMutangUniformMutation)
    assert sol._mutable_prefix == 1
    assert sol._seq_len == 2


def test_build_latent_sampling_hydramp_solver_pads_short_seed(
    fake_hf_model_with_decode: object,
    fake_tokenizer: object,
) -> None:
    """x0 length 1 < model sequence_length 2 triggers padding."""
    from unittest.mock import patch

    from amp_opt.latent_sampling_hydramp_init import build_latent_sampling_hydramp_solver
    from amp_opt.latent_sampling_solver import ProteinLatentSampling

    alphabet = ["<pad>", "A", "C", "D"]
    bb = _make_bb(alphabet, 1)
    x0 = np.array([["A"]], dtype=object)
    y0 = np.array([[0.0]])
    with (
        patch("transformers.AutoModel.from_pretrained", return_value=fake_hf_model_with_decode),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=fake_tokenizer),
    ):
        sol = build_latent_sampling_hydramp_solver(
            bb, x0, y0,
            {
                "hf_model_repo_id": "org/model",
                "hf_tokenizer_repo_id": "org/tok",
                "device": "cpu",
                "tau": 1.0,
            },
        )
    assert isinstance(sol, ProteinLatentSampling)
    assert sol._seq_len == 2


def test_build_brownian_hydramp_solver_returns_solver(
    fake_hf_model_with_decode: object,
    fake_tokenizer: object,
) -> None:
    from unittest.mock import patch

    from amp_opt.brownian_hydramp_init import build_brownian_hydramp_solver
    from amp_opt.brownian_solver import ProteinBrownianMotion

    alphabet = ["<pad>", "A", "C", "D"]
    bb = _make_bb(alphabet, 2)
    x0 = np.array([["A", "C"]], dtype=object)
    y0 = np.array([[0.0]])
    with (
        patch("transformers.AutoModel.from_pretrained", return_value=fake_hf_model_with_decode),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=fake_tokenizer),
    ):
        sol = build_brownian_hydramp_solver(
            bb, x0, y0,
            {
                "hf_model_repo_id": "org/model",
                "hf_tokenizer_repo_id": "org/tok",
                "device": "cpu",
                "sigma": 0.5,
            },
        )
    assert isinstance(sol, ProteinBrownianMotion)
    assert sol.sigma == 0.5


def test_brownian_solver_step_produces_valid_sequence(
    fake_hf_model_with_decode: object,
    fake_tokenizer: object,
) -> None:
    """Run 3 solver steps and verify outputs are valid AA sequences."""
    from unittest.mock import patch

    from amp_opt.brownian_hydramp_init import build_brownian_hydramp_solver

    alphabet = ["<pad>", "A", "C", "D"]
    bb = _make_bb(alphabet, 2)
    x0 = np.array([["A", "C"]], dtype=object)
    y0 = np.array([[0.0]])
    with (
        patch("transformers.AutoModel.from_pretrained", return_value=fake_hf_model_with_decode),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=fake_tokenizer),
    ):
        sol = build_brownian_hydramp_solver(
            bb, x0, y0,
            {
                "hf_model_repo_id": "org/model",
                "hf_tokenizer_repo_id": "org/tok",
                "device": "cpu",
                "sigma": 1.0,
            },
        )

    valid_tokens = {"A", "C"}
    for _ in range(3):
        x, y = sol.step()
        assert x.shape == (1, 2)
        assert y.shape == (1, 1)
        for tok in x.reshape(-1):
            assert str(tok) in valid_tokens, f"Unexpected token {tok!r}"


def test_brownian_solver_sigma0_reconstructs(
    fake_hf_model_with_decode: object,
    fake_tokenizer: object,
) -> None:
    """With sigma=0 (no noise), the walk never moves from z_0."""
    from unittest.mock import patch

    from amp_opt.brownian_hydramp_init import build_brownian_hydramp_solver

    alphabet = ["<pad>", "A", "C", "D"]
    bb = _make_bb(alphabet, 2)
    x0 = np.array([["A", "C"]], dtype=object)
    y0 = np.array([[0.0]])
    with (
        patch("transformers.AutoModel.from_pretrained", return_value=fake_hf_model_with_decode),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=fake_tokenizer),
    ):
        sol = build_brownian_hydramp_solver(
            bb, x0, y0,
            {
                "hf_model_repo_id": "org/model",
                "hf_tokenizer_repo_id": "org/tok",
                "device": "cpu",
                "sigma": 0.0,
            },
        )

    results = set()
    for _ in range(5):
        x, _y = sol.step()
        results.add(tuple(x.reshape(-1)))
    assert len(results) == 1, f"sigma=0 should produce identical outputs, got {results}"


def test_brownian_solver_max_step_norm_clamps(
    fake_hf_model_with_decode: object,
    fake_tokenizer: object,
) -> None:
    """Large sigma with tight max_step_norm should clamp the latent delta."""
    import torch
    from unittest.mock import patch

    from amp_opt.brownian_hydramp_init import build_brownian_hydramp_solver

    alphabet = ["<pad>", "A", "C", "D"]
    bb = _make_bb(alphabet, 2)
    x0 = np.array([["A", "C"]], dtype=object)
    y0 = np.array([[0.0]])
    max_norm = 0.01
    with (
        patch("transformers.AutoModel.from_pretrained", return_value=fake_hf_model_with_decode),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=fake_tokenizer),
    ):
        sol = build_brownian_hydramp_solver(
            bb, x0, y0,
            {
                "hf_model_repo_id": "org/model",
                "hf_tokenizer_repo_id": "org/tok",
                "device": "cpu",
                "sigma": 10.0,
                "max_step_norm": max_norm,
                "scaling_mode": "projected",
            },
        )

    parent = sol._pick_parent_row()
    input_ids = sol._bb_row_to_model_ids(parent)
    z_init = sol._encode(input_ids)
    sol._z = z_init.clone()

    _ = sol._next_candidate()
    z_after = sol._z
    delta_norm = float(torch.linalg.norm(z_after - z_init))
    assert delta_norm <= max_norm + 1e-6, f"Delta norm {delta_norm} exceeds max_step_norm {max_norm}"


@pytest.mark.parametrize("scaling_mode", ["riemannian", "projected", "metric"])
def test_brownian_solver_scaling_modes(
    fake_hf_model_with_decode: object,
    fake_tokenizer: object,
    scaling_mode: str,
) -> None:
    """Each scaling mode should produce valid token sequences."""
    from unittest.mock import patch

    from amp_opt.brownian_hydramp_init import build_brownian_hydramp_solver

    alphabet = ["<pad>", "A", "C", "D"]
    bb = _make_bb(alphabet, 2)
    x0 = np.array([["A", "C"]], dtype=object)
    y0 = np.array([[0.0]])
    with (
        patch("transformers.AutoModel.from_pretrained", return_value=fake_hf_model_with_decode),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=fake_tokenizer),
    ):
        sol = build_brownian_hydramp_solver(
            bb, x0, y0,
            {
                "hf_model_repo_id": "org/model",
                "hf_tokenizer_repo_id": "org/tok",
                "device": "cpu",
                "sigma": 0.1,
                "scaling_mode": scaling_mode,
            },
        )

    valid_tokens = {"A", "C"}
    x, y = sol.step()
    assert x.shape == (1, 2)
    assert y.shape == (1, 1)
    for tok in x.reshape(-1):
        assert str(tok) in valid_tokens, f"Unexpected token {tok!r} with scaling_mode={scaling_mode}"


def test_brownian_solver_second_order_runs(
    fake_hf_model_with_decode: object,
    fake_tokenizer: object,
) -> None:
    """second_order=True should not crash and produce valid output."""
    from unittest.mock import patch

    from amp_opt.brownian_hydramp_init import build_brownian_hydramp_solver

    alphabet = ["<pad>", "A", "C", "D"]
    bb = _make_bb(alphabet, 2)
    x0 = np.array([["A", "C"]], dtype=object)
    y0 = np.array([[0.0]])
    with (
        patch("transformers.AutoModel.from_pretrained", return_value=fake_hf_model_with_decode),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=fake_tokenizer),
    ):
        sol = build_brownian_hydramp_solver(
            bb, x0, y0,
            {
                "hf_model_repo_id": "org/model",
                "hf_tokenizer_repo_id": "org/tok",
                "device": "cpu",
                "sigma": 0.1,
                "second_order": True,
                "field_eps": 1e-3,
                "scaling_mode": "projected",
            },
        )

    valid_tokens = {"A", "C"}
    for _ in range(2):
        x, y = sol.step()
        assert x.shape == (1, 2)
        for tok in x.reshape(-1):
            assert str(tok) in valid_tokens, f"Unexpected token {tok!r} with second_order=True"
