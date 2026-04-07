from __future__ import annotations

import torch
import torch.nn as nn


class MusicTransformer(nn.Module):
    """
    Autoregressive Transformer for token-based music generation.

    Input tokens shape:  (batch, seq_len)
    Output logits shape: (batch, seq_len, vocab_size)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        max_seq_len: int = 256,
        num_genres: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.genre_embedding = nn.Embedding(num_genres, d_model)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, vocab_size)

    @staticmethod
    def causal_mask(seq_len: int, device: torch.device | str) -> torch.Tensor:
        """
        Additive attention mask with -inf above the main diagonal.
        """

        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask

    def forward(self, tokens: torch.Tensor, genre_ids: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            tokens: (batch, seq_len)
            genre_ids: (batch,) optional
        """

        batch_size, seq_len = tokens.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}")

        pos = torch.arange(seq_len, device=tokens.device).unsqueeze(0)  # (1, seq_len)
        x = self.token_embedding(tokens) + self.pos_embedding(pos)

        if genre_ids is None:
            genre_ids = torch.zeros(batch_size, dtype=torch.long, device=tokens.device)
        # Equivalent to h_t = Emb(x_t) + Emb(genre) at every time step.
        genre_cond = self.genre_embedding(genre_ids).unsqueeze(1).expand(-1, seq_len, -1)
        x = x + genre_cond

        x = self.dropout(x)
        
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=tokens.device)
        h = self.transformer(x, mask=mask, is_causal=True)  # (batch, seq_len, d_model)
        logits = self.head(h)
        return logits
