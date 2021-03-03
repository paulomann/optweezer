from torchvision.models import resnet 
import pytorch_lightning as pl
from typing import Literal, Dict
import torch.nn as nn
from torch.optim import Adam
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence
import collections



def init_weights(*models):
    for model in models:
        for name, param in model.named_parameters():
            if "weight" in name or "bias" in name:
                param.data.uniform_(-0.1, 0.1)

class MIL(pl.LightningModule):
    def __init__(self, optimizer_args: Dict[str, int], ftr_size: int = 126):
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

        self.save_hyperparameters()

    def configure_optimizers(self):
        # A gente pode botar o scheduler aqui também
        optimizer = Adam(self.parameters(), **self.optimizer_args)
        #optimizer = Adam(lr = 1e-3, betas = (0.9, 0.999), eps=1e-08, weight_decay=1e-5)

        return optimizer

    def forward(self, x, hiddens=None) -> torch.Tensor:

        if hiddens is None:
            hiddens = (torch.zeros(1,x.shape[0], self.ftr_size, device=x.device ),  torch.zeros(1,x.shape[0], self.ftr_size, device=x.device)) 
        
        _ , hiddens = self.lstm(x.reshape((10000, -1, 1)), hiddens)

        outs = self.classifier(hiddens[0]).squeeze()
        return outs, hiddens

    def training_step(self, train_batch, batch_idx, hiddens):
        x, y = train_batch

        logits, hiddens = self(x, hiddens)
        loss_fct = nn.CrossEntropyLoss()
        #loss = loss_fct(logits.view(-1, 2), y.view(-1))
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

        return {"loss": loss, "preds": preds, "targets": y, "hiddens": hiddens.detach()}

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
        logits, hiddens = self(x)

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

        return {"val_loss": loss, "preds": preds, "targets": y}

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