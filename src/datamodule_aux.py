import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import torch
import lightning.pytorch as pl
from torch.utils.data import random_split, DataLoader, Dataset
from typing import Optional
import os
import datetime

import rasterio, rasterio.plot
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.dataset import FDdataset
from configs.config import NUM_WORKERS


class FDdataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for flood detection using satellite imagery
    with auxiliary DEM and LULC data.

    Handles dataset splits, normalization, augmentation, and efficient data loading.
    All images, labels, and auxiliary data are preloaded into memory for fast access.

    Args:
        img_path (str): Path to training/validation/test images.
        pred_img_path (str): Path to prediction images.
        label_path (str): Path to segmentation labels.
        dem_path (str): Path to DEM files.
        lulc_path (str): Path to LULC files.
        split_path_train (str): File containing training split IDs.
        split_path_test (str): File containing test split IDs.
        split_path_val (str): File containing validation split IDs.
        split_path_pred (str): File containing prediction split IDs.
        batch_size (int): Batch size for dataloaders.

    Notes:
        - Assumes naming: <id>_image.tif, <id>_label.tif, <id>_image_dem.tif, <id>_image_lulc.tif
        - DEM is log-normalized (log1p) during caching.
        - Mean/std computed from training images only.
        - Augmentations are applied to image, DEM, and LULC jointly during training.
    """
    def __init__(self, 
                 img_path,
                 pred_img_path,
                 label_path,
                 dem_path,
                 lulc_path,
                 split_path_train, 
                 split_path_test, 
                 split_path_val, 
                 split_path_pred,
                 batch_size=16):
        super().__init__()
        
        self.img_path = img_path
        self.dem_path = dem_path
        self.lulc_path = lulc_path
        self.pred_img_path = pred_img_path
        self.label_path = label_path

        self.split_path_train = split_path_train
        self.split_path_test = split_path_test
        self.split_path_val = split_path_val
        self.split_path_pred = split_path_pred
        self.batch_size = batch_size
        self.train_transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5)
            # A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
            # A.ElasticTransform(p=0.3),
        ],
        additional_targets={
            "dem": "image",
            "lulc": "image"
        }
        )
        

    def setup(self, stage=None):
    
        # pre load all images into RAM here as a cache and pass to datasets
        # thus datasets would have o(1) lookup, reducing overhead

        with open(self.split_path_train) as f:
            self.train_data_file_names = f.read().splitlines()

        mean, std = self.compute_stats(self.train_data_file_names)

        print("Pre loading images into RAM...")
        all_files = set(self.train_data_file_names)
        
        # with open(self.split_path_train) as f:
        #     self.train_data_file_names = f.read().splitlines()
            
        # mean, std = self.compute_stats(self.train_data_file_names)
        # self.train_dataset = FDdataset(self.img_path, self.label_path, self.train_data_file_names, mean, std, transforms=self.train_transforms)

        with open(self.split_path_test) as f:
            test_file_names = f.read().splitlines()
        with open(self.split_path_val) as f:
            val_file_names = f.read().splitlines()
        with open(self.split_path_pred) as f:
            pred_file_names = f.read().splitlines()

        all_files.update(test_file_names, val_file_names, pred_file_names)

        self.test_data_file_names = test_file_names
        self.val_data_file_names = val_file_names
        self.pred_data_file_names = pred_file_names

        # for load data method
        _tmp = FDdataset(self.img_path, self.label_path, [], mean, std)
        
        image_cache = {}
        label_cache = {}
        dem_cache = {}
        lulc_cache = {}

        for file_name in all_files:
            mask = r"20240529_EO4_RES2_fl_pid_(\d+)"
            match = re.match(mask, file_name)
            
            if match:
                id_number = int(match.group(1))

            # ---------IMAGE-------
            if id_number >= 80:
                image_path = os.path.join(self.pred_img_path, file_name+'_image.tif')
            else:
                image_path = os.path.join(self.img_path, file_name+'_image.tif')
                
            img, temporal_coords, location_coords, meta = _tmp.load_data(image_path, None)

            image_cache[file_name + '_image.tif'] = {
                'img': img,
                'temporal_coords': temporal_coords,
                'location_coords': location_coords,
                'meta': meta
            }
            # --------LABEL---------
            label_path = os.path.join(self.label_path, file_name + '_label.tif')
            if os.path.exists(label_path):
                label_img, _, _ = self.read_tiff(label_path)
                label_img = label_img.squeeze(0)
                label_cache[file_name+'_label.tif'] = label_img

            # --------DEM---------
            dem_path = os.path.join(self.dem_path, file_name+'_image_dem.tif')
            if os.path.exists(dem_path):
                dem_img, _, _ = self.read_tiff(dem_path)
                dem_img = dem_img.squeeze(0)

                # normalize once (important)
                dem_img = np.log1p(dem_img)

                dem_cache[file_name + '_image_dem.tif'] = {
                    "data": dem_img.astype(np.float32)
                }
            
            #----------LULC---------
            lulc_path = os.path.join(self.lulc_path, file_name + '_lulc.tif')
            if os.path.exists(lulc_path):
                lulc_img, _, _ = self.read_tiff(lulc_path)
                lulc_img = lulc_img.squeeze(0).astype(np.int64)

                lulc_cache[file_name + '_image_lulc.tif'] = {
                    "data": lulc_img.astype(np.float32)
                }
            


        print(f"~~~~~~~ cached {len(image_cache)} images, {len(label_cache)} labels, {len(dem_cache)} dems, {len(lulc_cache)} lulc.")
        
        
        self.train_dataset = FDdataset(
            self.img_path,
            self.dem_path,
            self.lulc_path,
            self.label_path,
            self.train_data_file_names,
            mean,
            std,
            transforms=self.train_transforms,
            image_cache=image_cache,
            dem_cache=dem_cache,
            lulc_cache=lulc_cache,
            label_cache=label_cache
        )
        
        # mean, std = self.compute_stats(self.test_data_file_names)
        self.test_dataset = FDdataset(
            self.img_path,
            self.dem_path,
            self.lulc_path,
            self.label_path,
            self.test_data_file_names,
            mean,
            std,
            transforms=None,
            image_cache=image_cache,
            dem_cache=dem_cache,
            lulc_cache=lulc_cache,
            label_cache=label_cache
        )
        
        # mean, std = self.compute_stats(self.val_data_file_names)
        self.val_dataset = FDdataset(
            self.img_path,
            self.dem_path,
            self.lulc_path,
            self.label_path,
            self.val_data_file_names,
            mean,
            std,
            transforms=None,
            image_cache=image_cache,
            dem_cache=dem_cache,
            lulc_cache=lulc_cache,
            label_cache=label_cache
        )
        # self.pred_dataset = FDdataset()
    
        # mean, std = self.compute_stats(self.val_data_file_names)
        self.predict_dataset = FDdataset(
            self.img_path,
            self.dem_path,
            self.lulc_path,
            None,
            self.pred_data_file_names,
            mean,
            std,
            transforms=None,
            image_cache=image_cache,
            dem_cache=dem_cache,
            lulc_cache=lulc_cache,
            label_cache=label_cache
        )

    def compute_stats(self, file_names):
        mean = None
        std = None
        count = 0
        for file_name in file_names:
            path = os.path.join(self.img_path, file_name+'_image.tif')
            img, _, _ = self.read_tiff(path)
            img_mean = img.mean(axis=(1, 2))
            img_std = img.std(axis=(1, 2))

            if mean is None:
                mean = img_mean
                std = img_std
            else:
                mean += img_mean
                std += img_std
            
            count+=1
        
        mean /= count
        std /= count
        return mean, std
    
    def read_tiff(self, file_path):
        with rasterio.open(file_path) as src:
            img = src.read()
            meta = src.meta
            try:
                coords = src.lnglat()
            except:
                coords = None
        
        return img, meta, coords
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle=True, num_workers=NUM_WORKERS)
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = self.batch_size, num_workers=NUM_WORKERS)
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size = self.batch_size, num_workers=NUM_WORKERS)
    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size, num_workers=NUM_WORKERS)