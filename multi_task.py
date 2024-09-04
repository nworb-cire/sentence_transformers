import abc
import json

import pytorch_lightning as pl
import torch
from sentence_transformers import SentenceTransformer
from torch import nn
from torch.nn import functional as F


class ModuleWithLoss(pl.LightningModule, abc.ABC):
    @abc.abstractmethod
    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass


class LogisticRegression(ModuleWithLoss):
    def __init__(self, in_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)[:, 0]

    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(y_hat, y, reduction="none")


class NER(ModuleWithLoss):
    def __init__(self, in_features: int, num_labels: int):
        super().__init__()
        self.linear = nn.Linear(in_features, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(y_hat, y, reduction="none")


class MultiTask(ModuleWithLoss):
    def __init__(
        self,
        encoder: SentenceTransformer,
        heads: dict[str, tuple[ModuleWithLoss, float]],
        lr: float = 1e-3,
    ):
        """
        :param encoder: A SentenceTransformer model.
        :param heads: Dictionary task_name -> (head, weight). If weight is None, it is treated as 1.0.
        """
        super().__init__()
        self.encoder = encoder
        self.heads = nn.ModuleDict(
            {head: head_module for head, (head_module, _) in heads.items()}
        )
        self.weights = {head: weight for head, (_, weight) in heads.items()}
        self.lr = lr

    def forward(self, x: list[str]) -> dict[str, torch.Tensor]:
        x = self.encoder.encode(x, convert_to_tensor=True)
        return {head: self.heads[head](x) for head in self.heads}

    def loss(
        self, y_hat: dict[str, torch.Tensor], y: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return torch.mul(
            torch.cat(
                [
                    head.loss(y_hat[task], y[task]).unsqueeze(0)
                    for task, head in self.heads.items()
                ],
                dim=0,
            ),  # BxH
            torch.tensor([self.weights[head] for head in self.heads], device=self.device).unsqueeze(1),  # Hx1
        ).mean()

    def training_step(self, batch, batch_idx):
        x, *y = batch
        y_hat = self(x)
        y = {task: y[i] for i, task in enumerate(self.heads)}
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, *y = batch
        y_hat = self(x)
        y = {task: y[i] for i, task in enumerate(self.heads)}
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.heads.parameters(), lr=self.lr)


class DataModule(pl.LightningDataModule):
    def setup(self, stage: str) -> None:
        self.sentences = json.load(open("sentences.json"))
        self.sentiment_labels = torch.empty(
            len(self.sentences), dtype=torch.float
        ).random_(2)
        self.ner_labels = torch.empty(len(self.sentences), dtype=torch.long).random_(5)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            list(zip(self.sentences, self.sentiment_labels, self.ner_labels)),
            batch_size=4,
            shuffle=True,
        )


if __name__ == "__main__":
    encoder = SentenceTransformer("avsolatorio/GIST-small-Embedding-v0", revision=None)
    heads = {
        "sentiment": (
            LogisticRegression(encoder.get_sentence_embedding_dimension()),
            1.0,
        ),
        "ner": (NER(encoder.get_sentence_embedding_dimension(), 5), 0.5),
    }
    model = MultiTask(encoder, heads)
    model.to("cpu")  # for simplicity, we'll use CPU

    data_module = DataModule()
    trainer = pl.Trainer(fast_dev_run=True)
    trainer.fit(model, data_module)
    # TODO: save the model
