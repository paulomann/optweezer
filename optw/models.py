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



def init_weights(*models):
    for model in models:
        for name, param in model.named_parameters():
            if "weight" in name or "bias" in name:
                param.data.uniform_(-0.1, 0.1)

class MIL(pl.LightningModule):
    def __init__(self, optimizer_args: Dict[str, int], ftr_size: int = 126, bptt_steps: int = 100):
        super().__init__()
        self.optimizer_args = optimizer_args
 
        #self.lstm = nn.LSTM(ftr_size, ftr_size, 1, batch_first=False)


        self.lstm = nn.LSTM(1, ftr_size, 1, batch_first=False)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(ftr_size, 2)
        )

        self.ftr_size = ftr_size
        self.split_size = bptt_steps

        self.save_hyperparameters()

    def configure_optimizers(self):
        # A gente pode botar o scheduler aqui também
        optimizer = Adam(self.parameters(), **self.optimizer_args)
        #optimizer = Adam(lr = 1e-3, betas = (0.9, 0.999), eps=1e-08, weight_decay=1e-5)

        return optimizer


    def create_hiddens(self, x):
        
        if isinstance(x, torch.Tensor):
            batch_size = x.shape[0]
            device = x.device
        else:
            batch_size = x[0].shape[0]
            device = x[0].device

        hiddens = (torch.zeros(1, batch_size, self.ftr_size, device=device ),  
            torch.zeros(1, batch_size, self.ftr_size, device=device)) 
        
        return hiddens
                       
                       
    def forward(self, x) -> torch.Tensor:
        
        hiddens = self.create_hiddens(x)

        #if isinstance(x, torch.Tensor):
        #    _ , hiddens = self.lstm(x.reshape((x.shape[1], -1, 1)), hiddens)

        #else:
        #    for split in x:
        #        _ , hiddens = self.lstm(split.reshape((split.shape[1], -1, 1)), hiddens)
        #        hiddens = (hiddens[0].detach(), hiddens[1].detach())

        _ , hiddens = self.lstm(x.reshape((x.shape[1], -1, 1)), hiddens)
        logits = self.classifier(hiddens[0]).squeeze()

        return logits

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
        
    def tbptt_split_batch(self, batch, split_size):

        splits = []

        x = batch[0]
        y = batch[1]


        splits = [(x[:, i:i+split_size], y) for i in range (0, x.shape[-1], split_size)]

    

    #    for t in range(0, 10000, split_size):
    #        batch_split = []
    #        for i, x in enumerate(batch):
    #            if isinstance(x, torch.Tensor):
    #                split_x = x[:, t:t + split_size]
    #            elif isinstance(x, collections.Sequence):
    #                split_x = [None] * len(x)
    #                for batch_idx in range(len(x)):
    #                    split_x[batch_idx] = x[batch_idx][t:t + split_size]
#
    #            batch_split.append(split_x)
#
    #        splits.append(batch_split)

        return splits

    def init_layers(self):
        init_weights(self.resnet.fc)

    def freeze_layers(self):
        for child in self.resnet.children():
            for param in child.parameters():
                param.requires_grad = False

        for child in self.resnet.fc.children():
            for param in child.parameters():
                param.requires_grad = True