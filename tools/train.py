import argparse
import yaml
import os
import json
import numpy as np
from PIL import Image
import sys

from tqdm.auto import tqdm
import torch

sys.path.append(".")
from libs import utils
from libs.datasets.data_loader import DatasetLoader
from libs.models.aw_sdm import AWSDM
import libs.utils.inference as infer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True, type=str)
    parser.add_argument("--pretrained-path", required=True, type=str)
    parser.add_argument("--train-anno-path", required=True, type=str)
    parser.add_argument("--val-anno-path", type=str)
    parser.add_argument("--image-root", required=True, type=str)
    parser.add_argument("--feature-root", required=True, type=str)
    parser.add_argument("--feature-subdir-prefix", required=True, type=str)
    parser.add_argument("--save-path", default=".", type=str)
    parser.add_argument("--epoch", default=5, type=int)
    parser.add_argument("--order-th", default=0.1, type=float)
    parser.add_argument("--amodal-th", default=0.2, type=float)
    parser.add_argument("--dilate-kernel", default=0, type=int)
    args = parser.parse_args()
    return args


def main(args):
    with open(args.config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    for k, v in config.items():
        setattr(args, k, v)

    if not hasattr(args, "exp_path"):
        args.exp_path = os.path.dirname(args.config_path)

    model = AWSDM(args.model, dist_model=False)
    model.load_state(args.pretrained_path)
    model.switch_to("train")

    train_loader = DatasetLoader(
        args.train_anno_path, args.feature_root, args.feature_subdir_prefix
    )

    val_loader = DatasetLoader(
        args.val_anno_path, args.feature_root, args.feature_subdir_prefix
    )

    for epoch in tqdm(range(args.epoch), desc="Train"):
        loss = 0
        for data in tqdm(train_loader, leave=False):
            visible_mask, invisible_mask, final_mask, bbox, sd_feats = data
            model.set_input(rgb=sd_feats, mask=visible_mask, target=final_mask)
            loss = model.step()

        total_iou = 0
        for data in tqdm(val_loader, leave=False):
            visible_mask, invisible_mask, final_mask, bbox, sd_feats = data
            iou = model.evaluate(
                rgb=sd_feats, mask=visible_mask, bbox=bbox, target=final_mask
            )

            total_iou += iou

        mIoU = total_iou / val_loader.anno_len
        print(f"\nEpoch {epoch} mIou: {mIoU}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
