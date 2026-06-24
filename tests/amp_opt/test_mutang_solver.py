"""Tests for :class:`amp_opt.mutang_solver.ProteinMutangUniformMutation`."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import torch
from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.black_box_information import BlackBoxInformation

from amp_opt.mutang_solver import ProteinMutangUniformMutation, non_identity_mutation_pairs
from pep_compass_jr.tangent_space_mutation_proposer import substitutions_batch_from_jacobian


class _MockBlackBox(AbstractBlackBox):
    def __init__(self, *, alphabet: list[str], seq_len: int):
        super().__init__()
        self._info = BlackBoxInformation(
            "mock_mutang_solver",
            seq_len,
            True,
            True,
            True,
            alphabet,
        )

    def get_black_box_info(self) -> BlackBoxInformation:
        return self._info

    def _black_box(self, x: np.ndarray, context: object | None = None) -> np.ndarray:
        return np.zeros((x.shape[0], 1), dtype=np.float64)


def test_non_identity_mutation_pairs_filters_current() -> None:
    cur = np.array([0, 1], dtype=np.int64)
    d = {0: (0, 2), 1: (1, 2)}
    pairs = non_identity_mutation_pairs(d, cur)
    assert set(pairs) == {(0, 2), (1, 2)}


def test_next_candidate_single_hamming_distance_one() -> None:
    s_len, v_size, latent = 2, 3, 2
    ambient = s_len * v_size
    j_row = np.zeros((ambient, latent), dtype=np.float64)
    j_row[1, 0] = 1.0
    jac = np.stack([j_row], axis=0)

    alphabet = [f"T{i}" for i in range(v_size)]

    def encode(ids: torch.Tensor) -> torch.Tensor:
        assert ids.shape == (1, s_len)
        return torch.zeros((1, latent), dtype=torch.float32)

    def jac_fn(z: torch.Tensor) -> np.ndarray:
        assert z.shape == (1, latent)
        return jac

    bb = _MockBlackBox(alphabet=alphabet, seq_len=s_len)
    x0 = np.array([["T0", "T0"]], dtype=object)
    y0 = np.array([[1.0]])

    sol = ProteinMutangUniformMutation(
        bb,
        x0,
        y0,
        encode,
        jac_fn,
        sequence_length=s_len,
        vocab_size=v_size,
        min_number_of_directions=1,
        token_threshold=0.0,
        batch_size=3,
    )

    mut = sol.next_candidate()
    assert mut.shape == (3, s_len)
    for r in range(3):
        diff = sum(str(mut[r, j]) != str(x0[0, j]) for j in range(s_len))
        assert diff == 1


def test_n_mutations_repeated_jacobian_steps() -> None:
    s_len, v_size, latent = 2, 3, 2
    ambient = s_len * v_size
    j_row = np.zeros((ambient, latent), dtype=np.float64)
    j_row[1, 0] = 1.0
    jac = np.stack([j_row], axis=0)
    alphabet = [f"T{i}" for i in range(v_size)]

    def encode(ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros((1, latent), dtype=torch.float32)

    def jac_fn(z: torch.Tensor) -> np.ndarray:
        return jac

    bb = _MockBlackBox(alphabet=alphabet, seq_len=s_len)
    x0 = np.array([["T0", "T0"]], dtype=object)
    y0 = np.array([[1.0]])
    sol = ProteinMutangUniformMutation(
        bb,
        x0,
        y0,
        encode,
        jac_fn,
        sequence_length=s_len,
        vocab_size=v_size,
        min_number_of_directions=1,
        token_threshold=0.0,
        n_mutations=3,
    )
    rich = [{0: (0, 1, 2), 1: (0, 1, 2)}]
    calls = {"n": 0}

    def subst_fn(*args: object, **kwargs: object) -> list[dict[int, tuple[int, ...]]]:
        calls["n"] += 1
        return rich

    with patch("amp_opt.mutang_solver.substitutions_batch_from_jacobian", side_effect=subst_fn):
        mut = sol.next_candidate()
    assert mut.shape == (1, s_len)
    assert calls["n"] == 3


def test_tokenize_decode_round_trip_and_shape() -> None:
    bb_len, model_len, v_size, latent = 2, 4, 4, 1
    alphabet = [f"V{i}" for i in range(v_size)]
    jac = np.zeros((1, model_len * v_size, latent), dtype=np.float64)
    jac[0, 6, 0] = 1.0

    def encode(ids: torch.Tensor) -> torch.Tensor:
        assert ids.shape == (1, model_len)
        return torch.zeros((1, latent), dtype=torch.float32)

    def jac_fn(z: torch.Tensor) -> np.ndarray:
        return jac

    def tokenize_row(row: np.ndarray) -> torch.Tensor:
        m = {"V0": 1, "V1": 2}
        return torch.tensor(
            [[0, m[str(row[0])], m[str(row[1])], 3]],
            dtype=torch.long,
        )

    def decode_row(ids: np.ndarray) -> np.ndarray:
        inv = {1: "V0", 2: "V1"}
        return np.array(
            [inv[int(ids[1])], inv[int(ids[2])]],
            dtype=object,
        )

    bb = _MockBlackBox(alphabet=alphabet, seq_len=bb_len)
    x0 = np.array([["V0", "V0"]], dtype=object)
    y0 = np.array([[1.0]])

    sol = ProteinMutangUniformMutation(
        bb,
        x0,
        y0,
        encode,
        jac_fn,
        sequence_length=model_len,
        vocab_size=v_size,
        alphabet=alphabet,
        tokenize_row=tokenize_row,
        decode_row=decode_row,
        min_number_of_directions=1,
        token_threshold=0.0,
    )
    mut = sol.next_candidate()
    assert mut.shape == (1, bb_len)
    diff = sum(str(mut[0, j]) != str(x0[0, j]) for j in range(bb_len))
    assert diff >= 1


def test_raises_if_only_one_of_tokenize_decode() -> None:
    bb = _MockBlackBox(alphabet=["A", "B"], seq_len=1)

    def encode(ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros((1, 1))

    def jac_fn(z: torch.Tensor) -> np.ndarray:
        return np.zeros((1, 2, 1))

    with pytest.raises(ValueError, match="both be set"):
        ProteinMutangUniformMutation(
            bb,
            np.array([["A"]], dtype=object),
            np.array([[0.0]]),
            encode,
            jac_fn,
            sequence_length=1,
            vocab_size=2,
            tokenize_row=lambda r: torch.zeros((1, 1), dtype=torch.long),
            decode_row=None,
        )


def test_samples_drawn_only_from_jacobian_candidate_set() -> None:
    s_len, v_size, latent = 1, 3, 1
    ambient = s_len * v_size
    j_row = np.zeros((ambient, latent), dtype=np.float64)
    j_row[2, 0] = 1.0
    jac = np.stack([j_row], axis=0)

    alphabet = [f"T{i}" for i in range(v_size)]

    def encode(ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros((1, latent), dtype=torch.float32)

    def jac_fn(z: torch.Tensor) -> np.ndarray:
        return jac

    bb = _MockBlackBox(alphabet=alphabet, seq_len=s_len)
    x0 = np.array([["T0"]], dtype=object)
    y0 = np.array([[0.5]])
    sol = ProteinMutangUniformMutation(
        bb,
        x0,
        y0,
        encode,
        jac_fn,
        sequence_length=s_len,
        vocab_size=v_size,
        min_number_of_directions=1,
        token_threshold=0.0,
    )

    subst_kw = dict(
        sequence_length=s_len,
        vocab_size=v_size,
        min_number_of_directions=1,
        token_threshold=0.0,
    )
    expected_subst = substitutions_batch_from_jacobian(jac, **subst_kw)[0]
    cur = np.zeros((s_len,), dtype=np.int64)
    allowed = non_identity_mutation_pairs(expected_subst, cur)
    allowed_outcomes = {alphabet[new_id] for _pos, new_id in allowed}

    for _ in range(40):
        m = sol.next_candidate().reshape(-1)
        assert str(m[0]) in allowed_outcomes


def test_raises_when_no_non_identity_candidates() -> None:
    s_len, v_size = 1, 2
    alphabet = ["A", "B"]

    def encode(ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros((1, 1), dtype=torch.float32)

    def jac_fn(z: torch.Tensor) -> np.ndarray:
        return np.zeros((1, s_len * v_size, 1), dtype=np.float64)

    bb = _MockBlackBox(alphabet=alphabet, seq_len=s_len)
    x0 = np.array([["A"]], dtype=object)
    y0 = np.array([[0.0]])
    sol = ProteinMutangUniformMutation(
        bb,
        x0,
        y0,
        encode,
        jac_fn,
        sequence_length=s_len,
        vocab_size=v_size,
        min_number_of_directions=1,
        token_threshold=0.0,
    )

    with patch(
        "amp_opt.mutang_solver.substitutions_batch_from_jacobian",
        return_value=[{0: (0,)}],
    ):
        with pytest.raises(ValueError, match="no non-identity"):
            sol.next_candidate()


def test_raises_at_later_step_when_substitution_empty() -> None:
    s_len, v_size, latent = 1, 2, 1
    jac = np.zeros((1, s_len * v_size, latent), dtype=np.float64)

    def encode(ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros((1, 1), dtype=torch.float32)

    def jac_fn(z: torch.Tensor) -> np.ndarray:
        return jac

    bb = _MockBlackBox(alphabet=["A", "B"], seq_len=s_len)
    x0 = np.array([["A"]], dtype=object)
    y0 = np.array([[0.0]])
    sol = ProteinMutangUniformMutation(
        bb,
        x0,
        y0,
        encode,
        jac_fn,
        sequence_length=s_len,
        vocab_size=v_size,
        min_number_of_directions=1,
        token_threshold=0.0,
        n_mutations=2,
    )

    with patch(
        "amp_opt.mutang_solver.substitutions_batch_from_jacobian",
        side_effect=[[{0: (1,)}], [{}]],
    ):
        with pytest.raises(ValueError, match="mutation step 1"):
            sol.next_candidate()


def test_mutable_model_prefix_len_restricts_mutations() -> None:
    """With prefix=1, only position 0 can mutate even if Jacobian proposes position 1."""
    s_len, v_size, latent = 2, 3, 2
    bb_len = 1
    ambient = s_len * v_size
    j_row = np.zeros((ambient, latent), dtype=np.float64)
    j_row[1, 0] = 1.0  # position 0, vocab 1 — within prefix
    j_row[4, 0] = 1.0  # position 1, vocab 1 — outside prefix
    jac = np.stack([j_row], axis=0)

    alphabet = [f"T{i}" for i in range(v_size)]

    def encode(ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros((1, latent), dtype=torch.float32)

    def jac_fn(z: torch.Tensor) -> np.ndarray:
        return jac

    def tokenize_row(row: np.ndarray) -> torch.Tensor:
        flat = row.reshape(-1)
        ids = []
        for i in range(flat.size):
            ids.append(int(str(flat[i]).replace("T", "")))
        ids += [0] * (s_len - len(ids))
        return torch.tensor([ids], dtype=torch.long)

    def decode_row(ids_1d: np.ndarray) -> np.ndarray:
        return np.array([f"T{ids_1d[i]}" for i in range(bb_len)], dtype=object)

    bb = _MockBlackBox(alphabet=alphabet, seq_len=bb_len)
    x0 = np.array([["T0"]], dtype=object)
    y0 = np.array([[1.0]])

    sol = ProteinMutangUniformMutation(
        bb,
        x0,
        y0,
        encode,
        jac_fn,
        sequence_length=s_len,
        vocab_size=v_size,
        min_number_of_directions=1,
        token_threshold=0.0,
        batch_size=1,
        tokenize_row=tokenize_row,
        decode_row=decode_row,
        mutable_model_prefix_len=bb_len,
    )

    for _ in range(20):
        mut = sol.next_candidate()
        assert mut.shape == (1, bb_len)
        assert str(mut[0, 0]) != "T0"
        assert str(mut[0, 0]).startswith("T")
