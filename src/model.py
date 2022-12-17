from typing import Any, Optional


from transformers import AutoModelForSequenceClassification
from transformers import AutoConfig


from torchmetrics.functional import f1_score
import torch
import torch.nn.functional as F
from torch.optim import Optimizer
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from hydra_zen.typing import Partial


class EmotionModel(pl.LightningModule):
    def __init__(self, n_classes: int, hf_checkpoint: str, optimizer: Partial[Optimizer], lr: float) -> None:
        super().__init__()
        self.optimizer = optimizer
        self.learning_rate = lr
        self.n_classes = n_classes

        config = AutoConfig.from_pretrained(hf_checkpoint)
        config.num_labels = n_classes
        config.output_attentions = True

        self.model = AutoModelForSequenceClassification.from_pretrained(
            hf_checkpoint,
            config=config,
        )
        
        print(self.model)

    def configure_optimizers(self) -> Any:
        return self.optimizer(self.model.parameters(), lr=self.learning_rate)

    def training_step(self, batch, _) -> STEP_OUTPUT:
        input_ids, input_mask, labels = batch
        loss, logits, _ = self.model(
            input_ids=input_ids,
            attention_mask=input_mask,
            labels=labels,
            return_dict=False
        )

        preds = logits.argmax(dim=1)
        self.log("train_loss", loss)
        self.log("train_f1", f1_score(preds, labels, task="multiclass", num_classes=self.n_classes), prog_bar=True)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        return loss

    def validation_step(self, batch, _) -> STEP_OUTPUT:
        input_ids, input_mask, labels = batch
        loss, logits, _ = self.model(
            input_ids=input_ids,
            attention_mask=input_mask,
            labels=labels,
            return_dict=False
        )

        preds = logits.argmax(dim=1)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_f1", f1_score(preds, labels, task="multiclass", num_classes=self.n_classes), prog_bar=True)

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        return loss

    def test_step(self, batch, _) -> STEP_OUTPUT:
        input_ids, input_mask, labels = batch
        loss, logits, _ = self.model(
            input_ids=input_ids,
            attention_mask=input_mask,
            labels=labels,
            return_dict=False
        )

        preds = logits.argmax(dim=1)
        self.log("test_loss", loss)
        self.log("test_f1", f1_score(preds, labels, task="multiclass", num_classes=self.n_classes), prog_bar=True)

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        return loss
        