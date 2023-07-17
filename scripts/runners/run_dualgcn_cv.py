#!/usr/bin/env python
"""
Train and evaluate the DualGCN model.
"""

from __future__ import annotations

import typing as t

import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from tensorflow import keras

from cdrpy.models import dualgcn
from cdrpy.metrics import tf_metrics
from cdrpy.data.datasets import Dataset, EncodedDataset
from cdrpy.splits import load_splits


DegList = list[np.int32]
AdjList = list[list[int]]
DrugInput = t.Tuple[np.ndarray, DegList, AdjList]


data_folder = Path("data/inputs/GDSCv2DepMap")
dgcn_folder = data_folder / "DualGCN"
split_folder = Path(data_folder / "splits/tumor_blind")

drug_path = data_folder / "DrugToConvMolFeatures.pickle"
ppi_path = dgcn_folder / "MetadataPPIEdgeList.csv"
cnv_path = (
    dgcn_folder / "FeatureCellToCopyNumber689DualGCNGenesCNRatioLogp1.csv"
)
exp_path = dgcn_folder / "FeatureCellToExpression689DualGCNGenesTPMLogp1.csv"
label_path = data_folder / "LabelsLogIC50.csv"


def get_predictions_batches(
    model: keras.Model,
    batches: t.Generator[EncodedDataset, None, None],
    **metadata_args,
) -> pd.DataFrame:
    """"""
    results = []
    for batch_x, batch_y, batch_cell_ids, batch_drug_ids in batches:
        batch_x = [np.array(x) for x in batch_x]
        batch_preds = model.predict_on_batch(batch_x).reshape(-1)
        batch_res = pd.DataFrame(
            {
                "cell_id": batch_cell_ids,
                "drug_id": batch_drug_ids,
                "y_true": batch_y,
                "y_pred": batch_preds,
                **metadata_args,
            }
        )
        results.append(batch_res)

    return pd.concat(results)


def main() -> None:
    # GPU device memory issues on redwood5
    gpu_devices = tf.config.experimental.list_physical_devices("GPU")
    for gpu_device in gpu_devices:
        tf.config.experimental.set_memory_growth(gpu_device, True)

    drug_feat_encoder, drug_adj_encoder = dualgcn.load_drug_features(drug_path)
    cell_feat_encoder, cell_adj_encoder = dualgcn.load_cell_features(
        exp_path, cnv_path, ppi_path
    )

    # TODO: add scaling of the labels

    encoders = {
        "cell_encoders": [cell_feat_encoder, cell_adj_encoder],
        "drug_encoders": [drug_feat_encoder, drug_adj_encoder],
    }

    ds = Dataset.from_csv(label_path, name="gdsc_v2_depmap")
    split = list(load_splits(split_folder))[0]

    # for i, split in enumerate(load_splits(split_folder), 1):
    #     print(f"running for split {i}...")
    train_ds = ds.select(split.train_ids, name="train")
    val_ds = ds.select(split.val_ids, name="val")
    test_ds = ds.select(split.test_ids, name="test")

    cell_feat_norm = keras.layers.Normalization(axis=(1, 2))
    cell_feat_norm.adapt(np.array(cell_feat_encoder.encode(train_ds.cell_ids)))

    # NOTE: assumes the last axis is the feature axis
    cell_dim = cell_feat_encoder.shape[-1]
    drug_dim = drug_feat_encoder.shape[-1]

    print("creating model...")
    model = dualgcn.create_model(cell_dim, drug_dim, cell_feat_norm)
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=0.001,
            epsilon=None,
            decay=0.0,
            amsgrad=False,
        ),
        loss="mean_squared_error",
        metrics=["mse", tf_metrics.pearson],
    )

    print("encoding datasets...")
    train_ds_enc = train_ds.encode_tf(**encoders).shuffle(10000).batch(32)
    val_ds_enc = val_ds.encode_tf(**encoders).shuffle(10000).batch(32)

    print("training model...")
    hx = model.fit(train_ds_enc, epochs=1, validation_data=val_ds_enc)

    # pred_df = []
    # for d in (train_ds, val_ds, test_ds):
    #     gen = d.encode_batches(**encoders, batch_size=32, return_ids=True)
    #     preds = get_predictions_batches(model, gen, dataset=d.name, cv_fold=1)
    #     pred_df.append(preds)
    # pred_df = pd.concat(pred_df)

    # print(pred_df.head())

    # NOTE: CPU too slow (10+ mins)
    # TODO: try to write a train function so the encoded datasests get garbage collected
    #   since the get encoded in the train function.

    preds = model.predict(train_ds_enc)
    print(preds)

    # cpu_devices = tf.config.experimental.list_physical_devices("CPU")
    # if cpu_devices:
    #     with tf.device("CPU:0"):
    #         preds = model.predict(train_ds_enc)
    #         print(preds)

    # pred_df =

    # val_preds = get_predictions_batches(
    #     model,
    #     val_ds.encode_batches(**encoders, batch_size=32, return_ids=True),
    #     dataset=val_ds.name,
    #     cv_fold=1,
    # )

    # test_preds = get_predictions_batches(
    #     model,
    #     test_ds.encode_batches(**encoders, batch_size=32, return_ids=True),
    #     dataset=test_ds.name,
    #     cv_fold=1,
    # )

    # print(val_preds.head())

    # for batch_x, batch_y in train_ds_enc:
    #     batch_preds = model.predict_on_batch(batch_x)

    # for batch in val_ds.encode_batches(
    #     **encoders, batch_size=32, return_ids=True
    # ):
    #     batch_x, batch_y, batch_cell_ids, batch_drug_ids = batch
    #     batch_preds = model.predict_on_batch(batch_x).reshape(-1)
    #     batch_res = pd.DataFrame(
    #         {
    #             "cell_id": batch_cell_ids,
    #             "drug_id": batch_drug_ids,
    #             "y_true": batch_y,
    #             "y_pred": batch_preds,
    #             "split": val_ds.name,
    #             "fold": 0 + 1,
    #         }
    #     )


if __name__ == "__main__":
    main()
