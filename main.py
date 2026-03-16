import os
import lightning.pytorch as pl
import torch
from src.datamodule import FDdataModule
from src.model import model
from src.trainer import trainer

from configs.config import (
    IMG_PATH,
    PRED_IMG_PATH,
    LABEL_PATH,
    SPLIT_PATH_TRAIN,
    SPLIT_PATH_TEST,
    SPLIT_PATH_PRED,
    SPLIT_PATH_VAL,
    OUT_DIR
)


def main():
    print("Starting Flood Mapping Pipeline.")
    
    # Data
    print("Preparing Data")
    datamodule = FDdataModule(
        img_path=IMG_PATH,
        pred_img_path=PRED_IMG_PATH,
        label_path=LABEL_PATH,
        split_path_train=SPLIT_PATH_TRAIN,
        split_path_test=SPLIT_PATH_TEST,
        split_path_val=SPLIT_PATH_VAL,
        split_path_pred=SPLIT_PATH_PRED
    )

    # train
    print("Training ... ")
    trainer.fit(model, datamodule=datamodule)

    print("Testing ...")
    trainer.test(model, datamodule=datamodule)

    print("Predicting ... ")
    predictions = trainer.predict(model, datamodule=datamodule)

    save_predictions(predictions)

    print("Pipeline complete!")

def save_predictions(predictions):
    os.makedirs(OUT_DIR, exist_ok=True)

    for i, batch in enumerate(predictions):
        output_path = os.path.join(OUT_DIR, f"pred_{i}.pt")

        pl.utilities.rank_zero_info(f"Saving {output_path}")
        torch.save(batch, output_path)

if __name__ == "__main__":
    main()
