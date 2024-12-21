import argparse
import yaml
import os
import json
import numpy as np
from PIL import Image
import sys

from tqdm import tqdm
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
    parser.add_argument("--anno-path", required=True, type=str)
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
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    for k, v in config.items():
        setattr(args, k, v)

    if not hasattr(args, "exp_path"):
        args.exp_path = os.path.dirname(args.config)

    model = AWSDM(args.model, dist_model=False)
    model.load_state(args.pretrained_path)
    model.switch_to("train")

    train_loader = DatasetLoader(
        args.anno_path, args.feature_root, args.feature_subdir_prefix
    )

    for _ in tqdm(range(args.epoch)):
        for data in train_loader:
            visible_mask, invisible_mask, final_mask, bbox, sd_feats = data
            model.set_input(rgb=sd_feats, mask=visible_mask, target=final_mask)
            loss = model.step()
            print(loss["loss"])


if __name__ == "__main__":
    args = parse_args()
    main(args)
