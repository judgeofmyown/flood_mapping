import matplotlib.pyplot as plt
from terratorch.tasks import SemanticSegmentationTask
import numpy as np
import pandas as pd
import re
import torch
import lightning.pytorch as pl
import os
import datetime
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.callbacks import EarlyStopping
import rasterio, rasterio.plot
import albumentations as A
from albumentations.pytorch import ToTensorV2
from configs.config import SEED, OUT_DIR, EPOCHS


pl.seed_everything(SEED)

class PrintLossCallBack(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch

        train_loss = metrics.get('train/loss', 'N/A')
        val_loss = metrics.get('val/loss', 'N/A')

        if train_loss != 'N/A': train_los = round(train_loss.item(), 4)
        if val_loss != 'N/A': val_loss = round(val_loss.item(), 4)

        print(f"\n>>> Epoch {epoch} complete | Train Loss: {train_loss} | Val Loss: {val_loss}")

class progressBar(TQDMProgressBar):
    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.dynamic_ncols = False
        bar.ncols = 80
        return bar

logger = TensorBoardLogger(
    save_dir = OUT_DIR,
    name = "logs"
)

loggerWnB = WandbLogger(
    project="Flood_detection", # Name of your project on the website
    name="experiment-1-",               # Name of this specific run
    save_dir=OUT_DIR
)

checkpoint_callback = ModelCheckpoint(
    monitor="val/mIoU",
    mode="max",
    dirpath=os.path.join(OUT_DIR, "logs", "checkpoints"),
    filename="best-checkpoint-{epoch:02d}-{val_mIoU:.2f}",
    save_top_k=1
)

trainer = pl.Trainer(
    accelerator="gpu", # gpu enabled
    # strategy="auto",
    devices="1",
    precision="16-mixed",
    # fast_dev_run=True, ## used for testing purpose
    num_nodes=1,
    logger=logger,
    max_epochs=EPOCHS,
    check_val_every_n_epoch=1,
    log_every_n_steps=1,
    num_sanity_val_steps=0, # skipping datamodule sanity validation since takes time
    enable_checkpointing=True,
    enable_progress_bar=True,
    callbacks=[checkpoint_callback, progressBar(), PrintLossCallBack(), EarlyStopping(monitor="val/loss", mode="min", patience=5)],
)
