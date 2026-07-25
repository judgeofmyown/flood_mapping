"""
terratorch version
"""

import matplotlib.pyplot as plt
from terratorch.tasks import SemanticSegmentationTask
import numpy as np
import pandas as pd
import re
import torch
import lightning.pytorch as pl
import os
import datetime
import rasterio, rasterio.plot
import albumentations as A
from albumentations.pytorch import ToTensorV2
from configs.config import NUM_WORKERS, BANDS, NUM_FRAMES, HEAD_DROPOUT, LR, WEIGHT_DECAY, CLASS_WEIGHTS, BACKBONE_NAME
  


backbone_args = dict(
    backbone_pretrained = True,
    backbone = BACKBONE_NAME,
    backbone_bands = BANDS,
    backbone_num_frames = NUM_FRAMES,
    backbone_coords_encoding = ["location", "time"]
)

decoder_args = dict(
    decoder = "UperNetDecoder",
    decoder_channels = 256,
    decoder_scale_modules = True, 
)

necks = [
    dict(
        name = "SelectIndices",
        indices = [2, 5, 8, 11],
    ),
    dict(
        name="ReshapeTokensToImage",
        effective_time_dim=NUM_FRAMES,
    )
]

model_args = dict(
    **backbone_args,
    **decoder_args,
    num_classes=2,
    head_dropout=HEAD_DROPOUT,
    necks=necks,
    rescale=True,
)

model = SemanticSegmentationTask(
    model_args=model_args,
    plot_on_val=False,
    class_weights=None,
    loss="ce",
    lr=LR,
    optimizer="AdamW",
    optimizer_hparams=dict(weight_decay=WEIGHT_DECAY),
    ignore_index=-1,
    freeze_backbone=True,
    freeze_decoder=False,
    model_factory="EncoderDecoderFactory",
)