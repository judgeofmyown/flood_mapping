"""
terratorch version
"""

from terratorch.tasks import SemanticSegmentationTask
import numpy as np
import pandas as pd
import re
import torch
import lightning.pytorch as pl
import os
import datetime
import rasterio, rasterio.plot
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from terratorch.registry import TERRATORCH_HEAD_REGISTRY
from configs.config import NUM_WORKERS, BANDS, NUM_FRAMES, HEAD_DROPOUT, LR, WEIGHT_DECAY, CLASS_WEIGHTS, BACKBONE_NAME

class AuxEncoder(nn.Module):
    def __init__(self, in_channels=2, out_channels=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(),   
        )
    def forward(self, x):
        return self.encoder(x)

@TERRATORCH_HEAD_REGISTRY.register
class AuxFusionHead(nn.Module):
    def __init__(self, in_channels=256, aux_in_channels=2, aux_out_channels=64, num_classes=3, dropout=0.1):
        super().__init__()

        self.aux_encoder = AuxEncoder(aux_in_channels, aux_out_channels)

        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels + aux_out_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Dropout2d(dropout)
        )

        self.classifier = nn.Conv2d(in_channels, num_classes, kernel_size=1)
    
    def forward(self, x, aux=None):
        # x : (B, 256, H, W)
        # aux : (B, 2, H, W)
        if aux is not None:
            aux_feat = self.aux_encoder(aux)
        else:
            print("no aux data found")
        if aux_feat.shape[-2:] != x.shape[-2:]:
            aux_feat = F.interpolate(
                aux_feat, size=x.shape[-2:],
                mode='bilinear', align_corners=False
            )
        fused = torch.cat([x, aux_feat], dim=1)
        fused = self.fusion(fused)
        return self.classifier(fused)

class AuxEncoderDecoderWrapper(nn.Module):

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
    
    def forward(self, x, temporal_coords=None, location_coords=None, aux=None):
        encoder_output = self.base_model.encoder(
            x,
            temporal_coords=temporal_coords,
            location_coords=location_coords
        )
        neck_output = self.base_model.neck(encoder_output)
        decoder_output = self.base_model.decoder(neck_output) # [B, 256, H, W]

        head: AuxFusionHead = self.base_model.head
        return head(decoder_output, aux=aux)
    


class FloodSegmentationTask(SemanticSegmentationTask):
    
    def on_fit_start(self):
        self.model.model = AuxEncoderDecoderWrapper(self.model.model)
    
    def _common_step(self, batch):
        images = batch["image"]
        aux = batch["aux"]
        temporal = batch["temporal_coords"]
        location = batch["location_coords"]
        labels = batch["mask"]

        logits = self.model.model(
            images,
            temporal_coords=temporal,
            location_coords=location,
            aux=aux
        )
        return logits, labels
    
    def training_step(self, batch, batch_idx, dataloader_idx = 0):
        logits, labels = self._common_step(batch)
        loss = self.loss_fn(logits, labels)
        self.log("train/loss", loss, on_step=True, on_epoch=True)
        return super().training_step(batch, batch_idx, dataloader_idx)


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
    num_classes=3,
    head_dropout=HEAD_DROPOUT,
    necks=necks,
    rescale=True,
    head="AuxFusionHead",
    head_in_channels=256,
    head_aux_in_channels=2,
    head_aux_out_channels=64
)

model = FloodSegmentationTask(
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