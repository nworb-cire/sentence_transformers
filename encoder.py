import json

import torch
from torch import nn
from transformers import AutoTokenizer


class SentenceEncoder(nn.Module):
    def __init__(
        self,
        tokenizer: nn.Module,
        d_model: int = 768,
        num_layers: int = 4,
        nhead: int = 4,
        dim_feedforward: int = 3072,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.embedding = nn.Embedding(tokenizer.vocab_size, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, batch_first=True),
            num_layers=num_layers,
        )

    def forward(self, x: list[str]) -> torch.Tensor:
        x = self.tokenizer(x, return_tensors="pt", padding=True, truncation=True)
        x = self.encoder(self.embedding(x["input_ids"]), src_key_padding_mask=~x["attention_mask"].to(torch.bool))
        return x[:, 0, :]  # return only CLS token


if __name__ == "__main__":
    sentences = json.load(open("sentences.json"))
    # using this tokenizer since I will be using the model in part 2
    tokenizer = AutoTokenizer.from_pretrained("avsolatorio/GIST-small-Embedding-v0")
    encoder = SentenceEncoder(tokenizer)
    x = encoder(sentences)
    print(x.shape)
