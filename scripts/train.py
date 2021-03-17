import torch.nn as nn
import torch

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import seed_everything
from typing import Literal, List
import click

import optw
import optw.settings as settings
from optw.models import MIL
from optw.utils.logger import set_logger
from optw.data import ParticleDataset
from pathlib import Path

logger = set_logger(Path(__file__).name, verbose=settings.verbose)


@click.command()
@click.option(
    "--gpu", required=True, help=f"ID of the GPU to run the training", type=click.INT
)
@click.option("--name", help=f"Name of the training (for wandb)", type=click.STRING)
@click.option("--bsz", default=16, help=f"Batch Size", type=click.INT)
@click.option("--epochs", default=15, help=f"Number of epochs", type=click.INT)
@click.option("--bptt-steps", default=None, help=f"Number of steps needed before propagating gradients", type=click.INT)
@click.option(
    "--gradient-clip-val", default=0.5, help=f"Norm to clip gradients", type=click.FLOAT
)
@click.option("--log-every-n-steps", default=2, help=f"Log every n steps", type=click.INT)
@click.option("--lr", default=1e-3, help=f"Learning Rate", type=click.FLOAT)
@click.option("--b1", default=0.9, help=f"AdamW b1 beta parameter", type=click.FLOAT)
@click.option("--b2", default=0.999, help=f"AdamW b2 beta parameter", type=click.FLOAT)
@click.option(
    "--eps",
    default=1e-6,
    help=f"Adam's epsilon for numerical stability",
    type=click.FLOAT,
)
@click.option(
    "--weight-decay", default=0, help=f"Decoupled weight decay to apply", type=click.FLOAT
)
@click.option(
    "--correct-bias/--no-correct-bias",
    is_flag=True,
    help=f"Whether ot not to correct bias in Adam",
    default=True,
)
@click.option(
    "--wandb/--no-wandb",
    is_flag=True,
    help=f"Whether to use wandb or not",
    default=True,
)
@click.option(
    "--shuffle/--no-shuffle",
    is_flag=True,
    help=f"Whether to shuffle dataset's dataloader or not.",
    default=False,
)
@click.option(
    "--seed",
    default=42,
    help=f"Fix the seed of the random number generator.",
    type=click.INT,
)
@click.option(
    "--overfit",
    help=f"Overfit the dataset in a predetermined number of batches",
    type=click.INT,
    default=0
)
@click.option(
    "--model",
    help=f"Model to use for training. Use one of the follow: baseline; lstm; convolutional; hopfield",
    type=click.STRING,
    default='baseline'
)
def train(
    gpu: int,
    name: str,
    bsz: int,
    epochs: int,
    bptt_steps: int,
    gradient_clip_val: float,
    log_every_n_steps: int,
    lr: float,
    b1: float,
    b2: float,
    eps: float,
    weight_decay: float,
    correct_bias: bool,
    wandb: bool,
    shuffle: bool,
    seed: int,
    overfit: int,
    model: str
):
    seed_everything(seed)
    parameters = locals()
    seq_len = 10000
    train_dataloader = DataLoader(ParticleDataset('train'), batch_size= bsz, shuffle = shuffle)
    val_dataloader = DataLoader(ParticleDataset('val'), batch_size= bsz, shuffle = False)
    test_dataloader = DataLoader(ParticleDataset('test'), batch_size= bsz, shuffle = False)

    settings.MAX_SEQ_LENGTH = seq_len
    gradient_accumulation_steps = 1
    t_total = (len(train_dataloader) // gradient_accumulation_steps) * epochs
    optimizer_args = {
        "lr": lr,
        "betas": (b1, b2),
        "eps": eps,
        "weight_decay": weight_decay
        #"correct_bias": correct_bias
    }
    parameters.update(optimizer_args)
    logger.info(f"====> Parameters: {parameters}")
    wandb_logger = WandbLogger(project="optweezer", name=name, config=parameters)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
    )
    if not wandb:
        wandb_logger = None

    if model == 'baseline':
        ftr_size = train_dataloader.dataset.sequence_length
    else:
        ftr_size = 126

    ml_model = MIL(
        optimizer_args=optimizer_args,
        model = model,
        ftr_size= ftr_size,
        bptt_steps = bptt_steps
    )
    trainer = pl.Trainer(
        deterministic=True,
        max_steps=t_total,
        logger=wandb_logger if wandb else None,
        gpus=[gpu],
        accumulate_grad_batches=gradient_accumulation_steps,
        track_grad_norm=2,
        gradient_clip_val=gradient_clip_val,
        log_every_n_steps=log_every_n_steps,
        checkpoint_callback=checkpoint_callback,
        default_root_dir="models",
        overfit_batches=overfit,
        num_sanity_val_steps= 1
        
    )
    
    trainer.fit(ml_model, train_dataloader, val_dataloader)
    trainer.test(test_dataloaders=test_dataloader)


if __name__ == "__main__":
    train()