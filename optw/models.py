from torchvision.models import resnet 
import pytorch_lightning as pl
from typing import Literal, Dict
import torch.nn as nn
from torch.optim import Adam
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence
import collections

import wandb
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from optw.layers import Classifier, MLP, LSTM


def init_weights(*models):
    for model in models:
        for name, param in model.named_parameters():
            if "weight" in name or "bias" in name:
                param.data.uniform_(-0.1, 0.1)

class MIL(pl.LightningModule):
    def __init__(self, optimizer_args: Dict[str, int], model: Literal["baseline", "lstm", "hopfield"], ftr_size: int = 126, bptt_steps: int = 100):
        super().__init__()
        self.optimizer_args = optimizer_args



        if model == 'baseline':
            self.model = MLP(ftr_size, hidden_layers=512)
        elif model == 'lstm':
            self.model = LSTM(ftr_size)
            
        init_weights(self.model)
        self.save_hyperparameters()

    def configure_optimizers(self):
        # A gente pode botar o scheduler aqui também
        optimizer = Adam(self.parameters(), **self.optimizer_args)

        return optimizer
     
    def forward(self, x) -> torch.Tensor:
        return self.model(x).squeeze()

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch

        #splits = [x[:, i:i+self.split_size] for i in range (0, x.shape[-1], self.split_size)]
        #logits = self(splits)
        
        logits = self(x)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct( logits.view(-1, 2), y.view(-1))
        with torch.no_grad():

            preds = F.softmax(logits, dim=-1).argmax(dim=-1)
            acc = ((preds == y).sum().float()) / len(y)
            print()
            print(f"===>TRAIN PREDS: {preds}")
            print(f"===>TRAIN LABEL: {y}")
            print(f"===>TRAIN ACCUR: {acc}")
            print()
        preds = preds.unsqueeze(0) if preds.dim() == 0 else preds

        return {"loss": loss, "preds": preds, "targets": y}

    def training_epoch_end(self, outs):
        preds = []
        targets = []
        losses = []
        for out in outs:
            losses.append(out["loss"] * len(out["targets"]))
            targets.append(out["targets"])
            preds.append(out["preds"])

        preds = torch.cat(preds)
        targets = torch.cat(targets)
        loss = sum(losses) / len(targets)
        acc = ((preds == targets).sum().float()) / len(targets)
        print()
        print(f"===>TRAIN BATCH ACCUR: {acc}")
        print()
        self.log_dict({"train_loss": loss, "train_acc": acc})

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self(x)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, 2), y.view(-1))
        with torch.no_grad():

            preds = F.softmax(logits, dim=-1).argmax(dim=-1)
            acc = ((preds == y).sum().float()) / len(y)
            print()
            print(f"===>TRAIN PREDS: {preds}")
            print(f"===>TRAIN LABEL: {y}")
            print(f"===>TRAIN ACCUR: {acc}")
            print()
        
        self.log("val_acc", acc)
        return {"val_acc": acc, "val_loss": loss, "preds": preds, "targets": y}

    def validation_epoch_end(self, outs):
        
        preds = []
        targets = []
        losses = []
        for out in outs:
            losses.append(out["val_loss"] * len(out["targets"]))
            targets.append(out["targets"])
            preds.append(out["preds"])
        preds = torch.cat(preds)
        targets = torch.cat(targets)
        loss = sum(losses) / len(targets)
        acc = ((preds == targets).sum().float()) / len(targets)
        print()
        print(f"===>VAL BATCH ACCUR: {acc}")
        print()
        self.log_dict({"val_loss": loss, "val_acc": acc})


    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        logits = self(x)
        loss_fct = nn.CrossEntropyLoss()
        preds = F.softmax(logits, dim=-1).argmax(dim=-1)
        probas = F.softmax(logits, dim=-1)
        loss = loss_fct(logits.view(-1, 2), y.view(-1)).cpu()
        y = y.cpu().tolist()
        probas = probas.cpu().tolist()
        preds = preds.cpu().tolist()
        return (y, probas, preds, loss)

    def test_epoch_end(self, outputs):
        labels = []
        probas = []
        preds = []
        losses = []
        source_indices = []
        target_indices = []
        for i in outputs:
            labels.extend(i[0])
            probas.extend(i[1])
            preds.extend(i[2])

        self.logger.experiment.log(
            {
                "roc": wandb.plots.ROC(
                    np.array(labels), np.array(probas), ["Negative", "Positive"]
                )
            }
        )
        self.logger.experiment.log(
            {
                "cm": wandb.sklearn.plot_confusion_matrix(
                    np.array(labels), np.array(preds), ["Negative", "Positive"]
                )
            }
        )
        precision, recall, fscore, _ = precision_recall_fscore_support(
            labels, preds, average="binary"
        )
        self.log_dict({"precision": precision, "recall": recall, "fscore": fscore})
        
    #def tbptt_split_batch(self, batch, split_size):

    #    splits = []
    #    x = batch[0]
    #    y = batch[1]

    #    splits = [(x[:, i:i+split_size], y) for i in range (0, x.shape[-1], split_size)]
    #    return splits


