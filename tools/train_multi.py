import os
import sys
import yaml

sys.path.append(".")

import torch
import torch.distributed as dist

from libs.datasets.data_loader import DatasetLoader
from libs.models.aw_sdm import AWSDM


config = {
    "config_path": os.getenv("CONFIG_PATH"),
    "pretrained_path": os.getenv("PRETRAINED_PATH"),
    "train_anno_path": os.getenv("TRAIN_ANNO_PATH"),
    "val_anno_path": os.getenv("VAL_ANNO_PATH"),
    "image_root": os.getenv("IMAGE_ROOT"),
    "feature_root": os.getenv("FEATURE_ROOT"),
    "feature_subdir_prefix": os.getenv("FEATURE_SUBDIR_PREFIX"),
    "save_dir": os.getenv("SAVE_DIR"),
    "epoch": os.getenv("EPOCH"),
}


def train(rank, world_size):
    """Training function for each GPU process.

    Args:
        rank (int): The rank of the current process (one per GPU).
        world_size (int): Total number of processes.
    """
    # Initialize the process group for distributed training
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    with open(config["config_path"]) as f:
        config_yaml = yaml.load(f, Loader=yaml.FullLoader)
    # print(config_yaml)
    model = AWSDM(
        params=config_yaml,
        pretrained_path=config["pretrained_path"],
        dist_model=True,
        rank=rank,
    )
    model.switch_to("train")

    train_loader = DatasetLoader(
        config["train_anno_path"],
        config["feature_root"],
        config["feature_subdir_prefix"],
    )

    val_loader = DatasetLoader(
        config["val_anno_path"],
        config["feature_root"],
        config["feature_subdir_prefix"],
    )

    for epoch in range(int(config["epoch"])):
        for i, data in enumerate(train_loader):
            visible_mask, invisible_mask, final_mask, bbox, sd_feats = data
            model.set_input(
                rgb=sd_feats, mask=visible_mask, target=final_mask, rank=rank
            )
            loss = model.step()
            if i % config_yaml["trainer"]["print_freq"] == 0:
                print(f"Epoch: {epoch}, step: {i+1}, loss: {loss}")

        total_iou = 0
        for data in val_loader:
            visible_mask, invisible_mask, final_mask, bbox, sd_feats = data
            iou = model.evaluate(
                rgb=sd_feats,
                mask=visible_mask,
                bbox=bbox,
                target=final_mask,
                rank=rank,
            )

            total_iou += iou

        mIoU = total_iou / val_loader.anno_len
        print(f"\nEpoch: {epoch}, mIou: {mIoU}")

    # Clean up
    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    local_rank = int(os.environ["LOCAL_RANK"])
    train(local_rank, world_size)
