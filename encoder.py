import torch
from torch import nn


class SentenceEncoder(nn.Module):
    def __init__(
        self,
        encoder: nn.Module | None,
        tokenizer: nn.Module,
    ):
        super().__init__()
        if encoder is None:
            self.encoder = self._default_encoder()
        else:
            self.encoder = encoder
        self.tokenizer = tokenizer

    @staticmethod
    def _default_encoder():
        """Default BERT-like encoder"""
        n_layers = 12
        n_head = 4
        d_model = 768
        d_ff = 3072
        return nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_head, d_ff),
            n_layers
        )

    def forward(self, x: list[str]) -> torch.Tensor:
        x = self.tokenizer(x, return_tensors="pt", padding=True, truncation=True)
        x = self.encoder(x)
        return x
