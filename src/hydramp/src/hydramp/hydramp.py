"""HydrAMP neural network building blocks."""

from __future__ import annotations

import torch
from torch import nn


class HydrAMPGRU(nn.Module):
    """GRU cell stack used in the original HydrAMP implementation."""

    def __init__(
        self,
        units: int = 66,
        input_units: int = 66,
        output_len: int = 25,
    ) -> None:
        super().__init__()
        self.output_len = output_len
        self.units = units
        self.input_units = input_units

        self.kernel = nn.Parameter(torch.zeros(size=(input_units, units * 3)))
        self.recurrent_kernel = nn.Parameter(torch.zeros(size=(units, units * 3)))
        self.bias = nn.Parameter(torch.zeros(size=(units * 3,)))

    def _init_input(self, state: torch.Tensor) -> torch.Tensor:
        return torch.zeros((state.shape[0], self.input_units), device=state.device, dtype=state.dtype)

    def _init_state(self, input_: torch.Tensor) -> torch.Tensor:
        return torch.zeros((input_.shape[0], self.units), device=input_.device, dtype=input_.dtype)

    def cell_forward(self, inputs: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One recurrent update step."""
        matrix_x = torch.matmul(inputs, self.kernel) + self.bias
        x_z, x_r, x_h = torch.split(matrix_x, self.units, dim=-1)

        recurrent_matrix = torch.matmul(state, self.recurrent_kernel[:, : self.units * 2])
        recurrent_z, recurrent_r = torch.split(recurrent_matrix, self.units, dim=-1)

        z = torch.sigmoid(x_z + recurrent_z)
        r = torch.sigmoid(x_r + recurrent_r)
        recurrent_h = torch.matmul(r * state, self.recurrent_kernel[:, 2 * self.units :])

        h_tilde = torch.tanh(x_h + recurrent_h)
        hidden = z * state + (1 - z) * h_tilde
        return hidden, hidden

    def forward(self, input_: torch.Tensor | None, state: torch.Tensor | None = None) -> torch.Tensor:
        """Unroll recurrently for `output_len` steps."""
        if input_ is None and state is None:
            raise ValueError("Either input_ or state must be provided.")
        if input_ is None:
            assert state is not None
            input_ = self._init_input(state)
        if state is None:
            state = self._init_state(input_)

        current_output = input_
        current_state = state
        outputs: list[torch.Tensor] = []
        for _ in range(self.output_len):
            current_output, current_state = self.cell_forward(current_output, current_state)
            outputs.append(current_output)
        return torch.stack(outputs, dim=1)

    def forward_on_sequence(self, input_: torch.Tensor, state: torch.Tensor | None = None) -> torch.Tensor:
        """Run recurrence across sequence inputs of shape [B, T, D]."""
        if state is None:
            state = self._init_state(input_[:, 0])
        current_state = state
        outputs: list[torch.Tensor] = []
        for i in range(input_.shape[1]):
            current_output, current_state = self.cell_forward(input_[:, i], current_state)
            outputs.append(current_output)
        return torch.stack(outputs, dim=1)


class HydrAMPDecoder(nn.Module):
    """HydrAMP decoder that maps latent+condition vectors to token logits."""

    def __init__(
        self,
        sequence_length: int = 25,
        latent_dim: int = 64,
        condition_dim: int = 2,
        hidden_size: int = 100,
        vocab_size: int = 21,
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.vocab_size = vocab_size

        self.gru = HydrAMPGRU(
            units=latent_dim + condition_dim,
            input_units=latent_dim + condition_dim,
            output_len=sequence_length,
        )
        self.lstm = nn.LSTM(latent_dim + condition_dim + condition_dim, hidden_size, batch_first=True)
        self.dense = nn.Linear(hidden_size + condition_dim, vocab_size)

    def forward(
        self,
        latent_and_condition: torch.Tensor,
        return_logits: bool = True,
        gumbel_temperature: float = 1e-3,
    ) -> torch.Tensor:
        """Decode latent vectors (with condition appended) to sequence logits."""
        gru_output = self.gru(None, latent_and_condition)
        condition = latent_and_condition[:, -self.condition_dim :]
        condition_repeat = condition.unsqueeze(1).repeat(1, self.sequence_length, 1)
        gru_output_with_condition = torch.cat([gru_output, condition_repeat], dim=-1)
        lstm_output = self.lstm(gru_output_with_condition)[0]
        lstm_output_with_condition = torch.cat([lstm_output, condition_repeat], dim=-1)
        dense_output = self.dense(lstm_output_with_condition)
        if return_logits:
            return dense_output
        return torch.nn.functional.gumbel_softmax(dense_output, tau=gumbel_temperature)


class HydrAMPEncoder(nn.Module):
    """HydrAMP encoder that predicts latent mean and log_std from sequence tokens."""

    def __init__(
        self,
        vocab_size: int = 21,
        embedding_dim: int = 100,
        latent_dim: int = 64,
        sequence_length: int = 25,
        gru_hidden_size: int = 128,
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.gru1_f = HydrAMPGRU(input_units=embedding_dim, units=gru_hidden_size, output_len=sequence_length)
        self.gru1_r = HydrAMPGRU(input_units=embedding_dim, units=gru_hidden_size, output_len=sequence_length)
        self.gru2_f = HydrAMPGRU(input_units=gru_hidden_size * 2, units=gru_hidden_size, output_len=sequence_length)
        self.gru2_r = HydrAMPGRU(input_units=gru_hidden_size * 2, units=gru_hidden_size, output_len=sequence_length)
        self.mean_linear = nn.Linear(gru_hidden_size * 2, latent_dim)
        self.log_std_linear = nn.Linear(gru_hidden_size * 2, latent_dim)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.embedding(input_ids)
        gru1_f_output = self.gru1_f.forward_on_sequence(embeddings)
        gru1_r_output = self.gru1_r.forward_on_sequence(torch.flip(embeddings, (1,)))
        gru_1_output = torch.cat([gru1_f_output, torch.flip(gru1_r_output, (1,))], dim=-1)
        gru2_f_output = self.gru2_f.forward_on_sequence(gru_1_output)
        gru2_r_output = self.gru2_r.forward_on_sequence(torch.flip(gru_1_output, (1,)))
        gru_2_output = torch.cat([gru2_f_output[:, -1], gru2_r_output[:, -1]], dim=-1)
        mean = self.mean_linear(gru_2_output)
        log_std = self.log_std_linear(gru_2_output)
        return mean, log_std

    def encode(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Alias for :meth:`forward`; use on the encoder submodule, not :class:`HydrAMPModel`."""
        return self.forward(input_ids)
