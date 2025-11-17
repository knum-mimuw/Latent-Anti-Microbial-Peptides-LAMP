import torch
from torch import nn


class HydrAMPEncoder(nn.Module):
    """Simple GRU-based encoder."""

    def __init__(
        self,
        vocab_size=21,
        embedding_dim=100,
        hidden_dim=128,
        latent_dim=64,
        num_layers=2,
        device="cpu",
    ):
        super().__init__()
        self.device = device
        self.embedding = nn.Embedding(vocab_size, embedding_dim, device=device)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            device=device,
        )
        self.mean_linear = nn.Linear(hidden_dim * 2, latent_dim, device=device)
        self.std_linear = nn.Linear(hidden_dim * 2, latent_dim, device=device)

    def forward(self, x):
        """Encode input sequence to latent representation.

        Args:
            x: Input token sequence of shape [batch_size, seq_len]

        Returns:
            mean: Latent mean of shape [batch_size, latent_dim]
            std: Latent log_std of shape [batch_size, latent_dim]
        """
        x = x.to(self.device)
        embeddings = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        output, hidden = self.gru(
            embeddings
        )  # output: [batch_size, seq_len, hidden_dim*2]

        # Use last hidden state from both directions
        # hidden shape: [num_layers*2, batch_size, hidden_dim]
        last_hidden = hidden[-2:, :, :]  # Last layer, both directions
        last_hidden = last_hidden.transpose(0, 1)  # [batch_size, 2, hidden_dim]
        last_hidden = last_hidden.reshape(
            last_hidden.shape[0], -1
        )  # [batch_size, hidden_dim*2]

        mean = self.mean_linear(last_hidden)
        std = self.std_linear(last_hidden)
        return mean, std


class HydrAMPDecoder(nn.Module):
    """Simple GRU-based decoder."""

    def __init__(
        self,
        vocab_size=21,
        embedding_dim=100,
        hidden_dim=128,
        latent_dim=64,
        num_layers=2,
        max_length=25,
        device="cpu",
    ):
        super().__init__()
        self.device = device
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding = nn.Embedding(vocab_size, embedding_dim, device=device)
        self.latent_proj = nn.Linear(latent_dim, hidden_dim, device=device)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            device=device,
        )
        self.output_proj = nn.Linear(hidden_dim, vocab_size, device=device)

    def forward(self, z, target=None):
        """Decode latent representation to sequence.

        Args:
            z: Latent vector of shape [batch_size, latent_dim]
            target: Optional target sequence for teacher forcing [batch_size, seq_len]

        Returns:
            logits: Output logits of shape [batch_size, seq_len, vocab_size]
        """
        z = z.to(self.device)
        batch_size = z.shape[0]

        # Project latent to initial hidden state
        h_0 = self.latent_proj(z)  # [batch_size, hidden_dim]
        h_0 = h_0.unsqueeze(0).repeat(
            self.gru.num_layers, 1, 1
        )  # [num_layers, batch_size, hidden_dim]

        if target is not None:
            # Teacher forcing: use target sequence
            target = target.to(self.device)
            embeddings = self.embedding(target)  # [batch_size, seq_len, embedding_dim]
            output, _ = self.gru(embeddings, h_0)  # [batch_size, seq_len, hidden_dim]
        else:
            # Autoregressive generation
            outputs = []
            current_input = torch.zeros(
                (batch_size, 1, self.embedding.embedding_dim), device=self.device
            )
            hidden = h_0

            for _ in range(self.max_length):
                output, hidden = self.gru(current_input, hidden)
                outputs.append(output)
                # Use output as next input (project to embedding space)
                current_input = self.embedding(self.output_proj(output).argmax(dim=-1))

            output = torch.cat(outputs, dim=1)  # [batch_size, max_length, hidden_dim]

        logits = self.output_proj(output)  # [batch_size, seq_len, vocab_size]
        return logits

    def generate(
        self,
        z,
        max_length=None,
        temperature=1.0,
        top_k=None,
        top_p=None,
        eos_token_id=None,
    ):
        """Generate sequence from latent representation.

        Args:
            z: Latent vector of shape [batch_size, latent_dim]
            max_length: Maximum sequence length (defaults to self.max_length)
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top-k most likely tokens
            top_p: If set, use nucleus sampling with this probability threshold
            eos_token_id: If set, stop generation when this token is sampled

        Returns:
            Generated token sequences of shape [batch_size, seq_len]
        """
        if max_length is None:
            max_length = self.max_length

        z = z.to(self.device)
        batch_size = z.shape[0]

        # Project latent to initial hidden state
        h_0 = self.latent_proj(z)  # [batch_size, hidden_dim]
        h_0 = h_0.unsqueeze(0).repeat(
            self.gru.num_layers, 1, 1
        )  # [num_layers, batch_size, hidden_dim]

        # Initialize with start token (0)
        current_input = torch.zeros(
            (batch_size, 1, self.embedding.embedding_dim), device=self.device
        )
        hidden = h_0
        generated_tokens = []

        # Generation loop
        for step in range(max_length):
            # Forward through GRU
            output, hidden = self.gru(
                current_input, hidden
            )  # output: [batch_size, 1, hidden_dim]

            # Get logits
            logits = self.output_proj(output).squeeze(1)  # [batch_size, vocab_size]

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Apply top-k filtering
            if top_k is not None:
                top_k = min(top_k, logits.size(-1))
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float("-inf")

            # Apply top-p (nucleus) filtering
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1
                )

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            # Sample from logits
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [batch_size, 1]
            generated_tokens.append(next_token.squeeze(1))  # [batch_size]

            # Check for EOS token
            if eos_token_id is not None:
                # Stop if all sequences have generated EOS
                if (next_token.squeeze(1) == eos_token_id).all():
                    break

            # Use generated token as next input
            current_input = self.embedding(next_token)  # [batch_size, 1, embedding_dim]

        # Stack tokens: [batch_size, seq_len]
        generated_sequence = torch.stack(generated_tokens, dim=1)
        return generated_sequence
