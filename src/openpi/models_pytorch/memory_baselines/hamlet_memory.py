from typing import Any

import torch
from torch import nn


class MomentTokenPool(nn.Module):
    """Learnable moment tokens appended to prefix embeddings."""

    def __init__(self, num_tokens: int, feature_dim: int) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.feature_dim = feature_dim
        self.tokens = nn.Parameter(torch.randn(num_tokens, feature_dim) * 0.02)

    def forward(self, batch_size: int) -> torch.Tensor:
        return self.tokens.unsqueeze(0).expand(batch_size, -1, -1)


class HistoryMemoryTransformer(nn.Module):
    """Causal transformer over moment-token history."""

    def __init__(self, feature_dim: int, num_heads: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        seq_len = sequence.shape[1]
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=sequence.device, dtype=sequence.dtype),
            diagonal=1,
        )
        return self.encoder(sequence, mask=causal_mask)


class HamletMemoryAdapter(nn.Module):
    """HAMLET-style moment-token history adapter with runtime-state support."""

    def __init__(
        self,
        feature_dim: int,
        num_moment_tokens: int = 4,
        history_length: int = 4,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.0,
        gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.num_moment_tokens = num_moment_tokens
        self.history_length = history_length
        self.history_transformer = HistoryMemoryTransformer(feature_dim, num_heads, num_layers, dropout)
        self.gate = nn.Parameter(torch.tensor(gate_init, dtype=torch.float32))

        self.register_buffer("history_buffer", None, persistent=False)
        self.register_buffer("history_count", torch.tensor(0), persistent=False)

    def reset_runtime_state(self) -> None:
        self.history_buffer = None
        self.history_count = torch.tensor(0, device=self.gate.device)

    def get_runtime_state(self) -> dict[str, Any]:
        return {
            "history_buffer": self.history_buffer,
            "history_count": self.history_count,
        }

    def set_runtime_state(self, state: dict[str, Any] | None) -> None:
        if state is None:
            self.reset_runtime_state()
            return
        self.history_buffer = state.get("history_buffer")
        self.history_count = state.get("history_count", torch.tensor(0, device=self.gate.device))

    def _append_history(self, current_moment_tokens: torch.Tensor) -> None:
        current_step = current_moment_tokens.detach().unsqueeze(1)
        if self.history_buffer is None:
            self.history_buffer = current_step
        elif int(self.history_count.item()) < self.history_length:
            self.history_buffer = torch.cat([self.history_buffer, current_step], dim=1)
        else:
            self.history_buffer = torch.cat([self.history_buffer[:, 1:], current_step], dim=1)
        next_count = min(int(self.history_count.item()) + 1, self.history_length)
        self.history_count = torch.tensor(next_count, device=current_moment_tokens.device)

    def get_memory_stats(self) -> dict[str, int]:
        return {
            "history_count": int(self.history_count.item()),
            "history_length": self.history_length,
            "num_moment_tokens": self.num_moment_tokens,
        }

    def forward(self, current_moment_tokens: torch.Tensor, update_memory: bool = True) -> torch.Tensor:
        if current_moment_tokens.ndim != 3:
            raise ValueError(f"Expected (batch, num_moment_tokens, dim), got {tuple(current_moment_tokens.shape)}")
        if current_moment_tokens.shape[1] != self.num_moment_tokens:
            raise ValueError(
                f"Expected {self.num_moment_tokens} moment tokens, got {current_moment_tokens.shape[1]}"
            )
        if self.history_buffer is not None and self.history_buffer.shape[0] != current_moment_tokens.shape[0]:
            self.reset_runtime_state()

        history_steps: list[torch.Tensor] = []
        if self.history_buffer is not None and int(self.history_count.item()) > 0:
            history_steps.append(self.history_buffer[:, : int(self.history_count.item())])
        history_steps.append(current_moment_tokens.unsqueeze(1))

        sequence = torch.cat(history_steps, dim=1)  # (B, steps, M, D)
        batch_size, num_steps, num_tokens, feature_dim = sequence.shape
        sequence = sequence.reshape(batch_size, num_steps * num_tokens, feature_dim)
        contextualized = self.history_transformer(sequence)
        contextualized_current = contextualized[:, -self.num_moment_tokens :, :]

        output = current_moment_tokens + torch.tanh(self.gate) * contextualized_current
        if update_memory:
            self._append_history(current_moment_tokens)
        return output
