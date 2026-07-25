import rasterio
import numpy as np
import pandas as pd
import os
from math import inf
import matplotlib.pyplot as plt
from rasterio.warp import reproject, Resampling

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# dem_merged_pth = "data\dem\DEM_merged.tif"
# lulc_pth = "data\lulc\LULC.tif"
# output_dem_dir = "aligned_dem"
# output_lulc_dir = "aligned_lulc"

# pred_dem_dir = "data\prediction\dem"
# pred_lulc_dir = "data\prediction\lulc"
image_dir       = os.path.join(BASE_DIR, "data", "image")
pred_image_dir  = os.path.join(BASE_DIR, "data", "prediction", "image")
dem_merged_pth  = os.path.join(BASE_DIR, "data", "dem", "DEM_merged.tif")
lulc_pth        = os.path.join(BASE_DIR, "data", "lulc", "LULC.tif")
output_dem_dir  = os.path.join(BASE_DIR, "data", "aligned_dem")
output_lulc_dir = os.path.join(BASE_DIR, "data", "aligned_lulc")
pred_dem_dir    = os.path.join(BASE_DIR, "data", "prediction", "dem")
pred_lulc_dir   = os.path.join(BASE_DIR, "data", "prediction", "lulc")



def get_dataset_bounds():
    min_lon = inf
    min_lat = inf
    max_lon = -inf
    max_lat = -inf
    for img in os.listdir(image_dir):
        with rasterio.open(image_dir+f"\{img}") as src:
            bounds = src.bounds
            print(src.crs)
            break
            min_lon = min(min_lon, bounds.left)
            min_lat = min(min_lat, bounds.bottom)
            max_lon = max(max_lon, bounds.right)
            max_lat = max(max_lat, bounds.top)

    for img in os.listdir(pred_image_dir):
        with rasterio.open(pred_image_dir+f"\{img}") as src:
            bounds = src.bounds
            print(src.crs)
            break
            min_lon = min(min_lon, bounds.left)
            min_lat = min(min_lat, bounds.bottom)
            max_lon = max(max_lon, bounds.right)
            max_lat = max(max_lat, bounds.top)
    print(f"{min_lon}, {min_lat}, {max_lon}, {max_lat}")
    return [min_lon, min_lat, max_lon, max_lat]

def dem_test():
    for dem_img in os.listdir(dem_merged_pth):
        with rasterio.open(os.path.join(dem_merged_pth, dem_img)) as src:
            print(src.dtypes)
            print(src.crs) # EPSG 4326
            print(src.nodata)
            print(src.read(1).min(), src.read(1).max())
            dem = src.read(1)
            dem_norm = (dem - np.nanmin(dem)) / (np.nanmax(dem) - np.nanmin(dem))

            plt.imshow(dem_norm, cmap="terrain")
            plt.colorbar()
            plt.show()
        break

def align_raster_to_image(ref_path, source_path, output_path, resampling=Resampling.bilinear):
    with rasterio.open(ref_path) as ref:
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_width = ref.width
        ref_height = ref.height

        with rasterio.open(source_path) as src:
            source_data = src.read(1)

            aligned = np.zeros((ref_height, ref_width), dtype=source_data.dtype)

            reproject(
                source=source_data,
                destination=aligned,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=resampling
            )
        
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=ref_height,
            width=ref_width,
            count=1,
            dtype=aligned.dtype,
            crs=ref_crs,
            transform=ref_transform,
        ) as dst:
            dst.write(aligned, 1)


def batch_generate_aux_data():
    
    """
    Generates all auxilary data corresponding to each provided image.
    """

    train_files = os.listdir(image_dir)
    pred_files = os.listdir(pred_image_dir)
    total = len(train_files) + len(pred_files)
    count = 0

    for img_file in train_files:
        count+=1
        image_pth = os.path.join(image_dir, img_file)
        output_dem_pth = os.path.join(output_dem_dir, img_file.replace(".tif", "_dem.tif"))
        output_lulc_pth = os.path.join(output_lulc_dir, img_file.replace(".tif", "_lulc.tif"))

        align_raster_to_image(image_pth, dem_merged_pth, output_dem_pth, Resampling.bilinear)
        align_raster_to_image(image_pth, lulc_pth, output_lulc_pth, Resampling.nearest)
        print(f"[{count}/{total}]  Train/Test | {img_file}")
        
    
    for img_file in pred_files:
        count+=1

        image_pth = os.path.join(pred_image_dir, img_file)
        output_dem_pth = os.path.join(pred_dem_dir, img_file.replace(".tif", "_dem.tif"))
        output_lulc_pth = os.path.join(pred_lulc_dir, img_file.replace(".tif", "_lulc.tif"))

        align_raster_to_image(image_pth, dem_merged_pth, output_dem_pth, Resampling.bilinear)
        align_raster_to_image(image_pth, lulc_pth, output_lulc_pth, Resampling.nearest)
        print(f"[{count}/{total}]  Pred | {img_file}")

# align_raster_to_image(r"data\image\20240529_EO4_RES2_fl_pid_001_image.tif", r"data\dem\DEM_merged.tif", r"data\aligned_dem\20240529_EO4_RES2_fl_pid_001_dem.tif")
# align_raster_to_image(r"data\image\20240529_EO4_RES2_fl_pid_001_image.tif", r"data\lulc\LULC.tif", r"data\aligned_lulc\20240529_EO4_RES2_fl_pid_001_lulc.tif")
# get_dataset_bounds()
# dem_test()

batch_generate_aux_data()
