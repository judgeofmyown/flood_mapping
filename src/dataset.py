import numpy as np
import pandas as pd
import re
import torch
from torch.utils.data import random_split, DataLoader, Dataset
from typing import Optional
import os
import datetime
import rasterio, rasterio.plot


class FDdataset(Dataset):
    def __init__(self, 
                 img_path, 
                 label_path, 
                 files, 
                 mean, 
                 std, 
                 transforms=None, 
                 image_cache=None, 
                 label_cache=None):
        super().__init__()

        self.img_path = img_path
        self.label_path = label_path
        self.files = files
        self.mean = mean
        self.std = std
        self.NO_DATA = np.nan
        self.NO_DATA_FLOAT = 0.00001
        self.transforms = transforms
        self.image_cache = image_cache
        self.label_cache = label_cache
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img_file_name = self.files[idx] + '_image.tif'
        img_file_path = os.path.join(self.img_path, img_file_name)

        # img, temporal_coords, location_coords, meta = self.load_data(img_file_path, None)
        if self.image_cache is not None:
            cached = self.image_cache[img_file_name]
            img = cached["img"]
            temporal_coords = cached["temporal_coords"]
            location_coords = cached["location_coords"]
        else:
            # img, temporal_coords, location_coords, meta = self.load_data(img_file_path, None)
            pass
        
        label_file_path = None
        if self.label_path is not None:
            label_file_name = self.files[idx] + '_label.tif'

            if self.label_cache is not None:
                label_img = self.label_cache[label_file_name]
            else:    
                # label_file_path = os.path.join(self.label_path, label_file_name)
                # label_img, _, _ = self.read_tiff(label_file_path)
                # label_img = label_img.squeeze(0)
                pass
        

        if self.label_path is not None:
            if self.transforms:
                # since for albumentations we need (W, H, C) shape and we have (C, 1, W, H)
                img_for_aug = img[:, 0, :, :] # -> (C ,W, H)
                img_for_aug = np.moveaxis(img_for_aug, 0, -1).astype(np.float32) # -> (W, H, C)
                augmented = self.transforms(image=img_for_aug, mask=label_img)
                img_for_aug = augmented["image"]
                label_img = augmented["mask"]
                # restore out shape of (1, C, W, H)
                img = np.expand_dims(np.moveaxis(img_for_aug, -1, 0), axis=1)

            return {
                "image": img,
                "temporal_coords": torch.tensor(temporal_coords, dtype=torch.float32),
                "location_coords": torch.tensor(location_coords, dtype=torch.float32),
                "mask": torch.tensor(label_img, dtype=torch.long)
            }
        else:
            return {
                "image": img,
                "temporal_coords": torch.tensor(temporal_coords, dtype=torch.float32),
                "location_coords": torch.tensor(location_coords, dtype=torch.float32)
            }
        
    def load_data(self, file_path, indices):
        temporal_coords = None
        img, meta, coords = self.read_tiff(file_path)
    
        img = np.moveaxis(img, 0, -1) # (C to last)
        if indices is not None:
            img = img[..., indices]
        img = np.where(img==self.NO_DATA, self.NO_DATA_FLOAT, (img - self.mean) / self.std)
    
        try:
            match = re.search(r'(\d{8})', file_path)
            if match:
                date_str = match.group(1)
                year = int(date_str[:4])
                julian_day = datetime.datetime.strptime(date_str, "%Y%m%d").timetuple().tm_yday
                temporal_coords = [[year, julian_day]]
        except Exception as e:
            print(f"Couln not extract timestamps for {file_path} ({e})")
        
        # imgs = np.stack(imgs, axis=0)
        img = np.expand_dims(img, axis=0)
        # imgs = np.moveaxis(imgs, -1, 0).astype('float32')
        img = np.moveaxis(img, -1, 0)
        # imgs = np.expand_dims(imgs, axis=0) # batch dimension
    
        return img, temporal_coords, coords, meta

    def read_tiff(self, file_path):
        with rasterio.open(file_path) as src:
            img = src.read()
            meta = src.meta
            try:
                coords = src.lnglat()
            except:
                coords = None
        
        return img, meta, coords
