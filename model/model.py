"""Format-conditioned image -> markup seq2seq.

A small CNN downsamples the grayscale image into a feature grid, which is
flattened (with a 2-D positional embedding) into a sequence of encoder memory
states. A Transformer decoder autoregressively emits the markup, cross-attending
to that memory. The format token (``[LATEX]`` / ``[TYPST]``) is just the first
decoder input symbol — no separate conditioning path is needed.

Everything is config-driven so the same code scales from the overfit smoke test
to a real run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 256
    nhead: int = 8
    num_decoder_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    pad_id: int = 0
    max_len: int = 1024


class CNNEncoder(nn.Module):
    """Grayscale image -> sequence of d_model feature vectors."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.cnn = nn.Sequential(
            block(1, 32),    # /2
            block(32, 64),   # /4
            block(64, 128),  # /8
            block(128, d_model),  # /16
        )
        # Learned 2-D positional embedding added over the feature grid.
        self.row_embed = nn.Parameter(torch.randn(64, d_model) * 0.02)
        self.col_embed = nn.Parameter(torch.randn(64, d_model) * 0.02)

    def forward(self, image: Tensor) -> Tensor:
        feat = self.cnn(image)  # (B, d, H', W')
        b, d, h, w = feat.shape
        feat = feat.permute(0, 2, 3, 1)  # (B, H', W', d)
        pos = self.row_embed[:h].unsqueeze(1) + self.col_embed[:w].unsqueeze(0)
        feat = feat + pos.unsqueeze(0)
        return feat.reshape(b, h * w, d)  # (B, H'*W', d)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[: x.size(1)].unsqueeze(0)


class Seq2Seq(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = CNNEncoder(cfg.d_model)
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.pos = PositionalEncoding(cfg.d_model, cfg.max_len)
        layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, cfg.num_decoder_layers)
        self.out = nn.Linear(cfg.d_model, cfg.vocab_size)

    def encode(self, image: Tensor) -> Tensor:
        return self.encoder(image)

    def decode_step(self, tgt: Tensor, memory: Tensor) -> Tensor:
        """tgt: (B, T) ids -> logits (B, T, V)."""
        t = tgt.size(1)
        x = self.pos(self.embed(tgt) * math.sqrt(self.cfg.d_model))
        causal = torch.triu(torch.ones(t, t, dtype=torch.bool, device=tgt.device), diagonal=1)
        pad_mask = tgt == self.cfg.pad_id
        h = self.decoder(x, memory, tgt_mask=causal, tgt_key_padding_mask=pad_mask)
        return self.out(h)

    def forward(self, image: Tensor, tgt_in: Tensor) -> Tensor:
        memory = self.encode(image)
        return self.decode_step(tgt_in, memory)

    @torch.no_grad()
    def greedy_decode(
        self, image: Tensor, format_id: int, bos_id: int, eos_id: int, max_len: int = 256
    ) -> list[int]:
        """Decode a single image (batch of 1) given a format token."""
        self.eval()
        memory = self.encode(image)
        seq = torch.tensor([[format_id, bos_id]], device=image.device)
        for _ in range(max_len):
            logits = self.decode_step(seq, memory)
            nxt = int(logits[0, -1].argmax())
            seq = torch.cat([seq, torch.tensor([[nxt]], device=image.device)], dim=1)
            if nxt == eos_id:
                break
        return seq[0].tolist()
