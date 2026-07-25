import os
from pathlib import Path

def get_root_dir() -> Path:
    return Path(__file__).parent.parent

BASE_DIR = get_root_dir()
os.mkdir("\output")
OUT_DIR = os.path.join(BASE_DIR, "\output")
EPOCHS              = 25
LR                  = 4.0e-5
WEIGHT_DECAY        = 0.1
HEAD_DROPOUT        = 0.3
FREEZE_BACKBONE     = False
CLASS_WEIGHTS       = [1, 1.5]

BANDS               = ['HH', 'HV', 'Green', 'Red', 'NIR', 'SWIR']
NUM_FRAMES          = 1
NUM_WORKERS         = 4

BACKBONE_NAME       = "prithvi_eo_v2_tiny_tl"

IMG_PATH            = os.path.join(BASE_DIR, "data\image")
PRED_IMG_PATH       = os.path.join(BASE_DIR, "data\prediction\image")
LABEL_PATH          = os.path.join(BASE_DIR, "data\label")
SPLIT_PATH_TRAIN    = os.path.join(BASE_DIR, "data\split\train.txt")
SPLIT_PATH_TEST     = os.path.join(BASE_DIR, "data\image\test.txt")
SPLIT_PATH_VAL      = os.path.join(BASE_DIR, "data\image\val.txt")
SPLIT_PATH_PRED     = os.path.join(BASE_DIR, "data\image\pred.txt")

SEED                = 0