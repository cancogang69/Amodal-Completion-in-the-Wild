import os
import torch
import torch.nn as nn
import numpy as np
import cvbase as cvb
import cv2

from libs.utils import mask_to_bbox, crop_padding


class DatasetLoader(object):
    def __init__(
        self, anno_path, feature_root, feature_subdir_prefix="t_181_index_"
    ):
        data = cvb.load(anno_path)
        images_info = dict(
            [[img_info["id"], img_info] for img_info in data["images"]]
        )
        annos_info = data["annotations"]

        self.annos = []
        for anno in annos_info:
            img_info = images_info[anno["image_id"]]
            black_start = 0
            black_end = 0
            if anno["last_col"] > 0:
                black_start = anno["last_col"]
                black_end = img_info["width"]
            else:
                black_end = anno["last_col"] + 1

            feature_file_name = (
                f"{img_info['file_name'].split('.')[0]}_{anno['id']}.pt"
            )

            self.annos.append(
                {
                    "mask": anno,
                    "image_file_name": img_info["file_name"],
                    "feature_file_name": feature_file_name,
                    "image_height": img_info["height"],
                    "image_width": img_info["width"],
                    "black_start": black_start,
                    "black_end": black_end,
                }
            )

        self.feature_root = feature_root
        self.feature_subdir_prefix = feature_subdir_prefix

    def __iter__(self):
        self.curr_idx = 0
        return self

    def __get_feature_from_save(self, feature_file_name):
        org_src_ft_dict = {}
        for layer_i in [0, 1, 2, 3]:
            feat_dir = os.path.join(
                self.feature_root,
                f"{self.feature_subdir_prefix}_{str(layer_i)}",
            )
            feat = torch.load(
                os.path.join(feat_dir, f"{feature_file_name.split('.')[0]}_.pt")
            )
            org_src_ft = feat.permute(1, 2, 0).float().numpy()  # h x w x L
            org_src_ft_dict[layer_i] = org_src_ft

        return org_src_ft_dict

    def __combime_mask_with_sd_features(
        self, image_height, image_width, bbox, sd_features
    ):
        org_h, org_w = image_height, image_width
        src_ft_dict = {}
        for layer_i in [0, 1, 2, 3]:
            org_src_ft = sd_features[layer_i]
            src_ft_new_bbox = [
                int(bbox[0] * org_src_ft.shape[1] / org_w),
                int(bbox[1] * org_src_ft.shape[0] / org_h),
                int(bbox[2] * org_src_ft.shape[1] / org_w),
                int(bbox[3] * org_src_ft.shape[0] / org_h),
            ]
            src_ft = crop_padding(
                org_src_ft,
                src_ft_new_bbox,
                pad_value=(0,) * org_src_ft.shape[-1],
            )
            src_ft = torch.tensor(src_ft).permute(2, 0, 1).unsqueeze(0)
            src_ft = src_ft.to("cuda")
            if layer_i == 0:
                cur_upsample_sz = 24
            elif layer_i == 1:
                cur_upsample_sz = 48
            else:
                cur_upsample_sz = 96
            if src_ft.shape[-2] != 0 and src_ft.shape[-1] != 0:
                src_ft = nn.Upsample(
                    size=(cur_upsample_sz, cur_upsample_sz), mode="bilinear"
                )(src_ft).squeeze(
                    0
                )  # L x h x w
                src_ft = src_ft.permute(1, 2, 0).cpu().numpy()  # h x w x L
            else:
                src_ft = torch.tensor(org_src_ft).permute(2, 0, 1).unsqueeze(0)
                src_ft = nn.Upsample(size=(org_h, org_w), mode="bilinear")(
                    src_ft
                ).squeeze(
                    0
                )  # L x h x w
                src_ft = src_ft.permute(1, 2, 0).cpu().numpy()  # h x w x L
                src_ft = crop_padding(
                    src_ft, bbox, pad_value=(0,) * src_ft.shape[-1]
                )  # h x w x L
                src_ft = torch.tensor(src_ft).permute(2, 0, 1).unsqueeze(0)
                print(cur_upsample_sz, cur_upsample_sz)
                src_ft = nn.Upsample(
                    size=(cur_upsample_sz, cur_upsample_sz), mode="bilinear"
                )(src_ft).squeeze(
                    0
                )  # L x h x w
                src_ft = src_ft.permute(1, 2, 0).cpu().numpy()  # h x w x L

            src_ft_dict[layer_i] = src_ft

        return src_ft_dict

    def __get_mask(self, height, width, polygons):
        mask = np.zeros([height, width])
        for polygon in polygons:
            mask = cv2.fillPoly(
                mask, np.array([polygon]), color=[255, 255, 255]
            )

        return mask

    def __next__(self):
        anno = self.annos[self.curr_idx]
        image_h, image_w = anno["image_height"], anno["image_width"]

        visible_mask = self.__get_mask(
            image_h, image_w, anno["mask"]["visible_segmentations"]
        )
        invisible_mask = self.__get_mask(
            image_h, image_w, anno["mask"]["invisible_segmentations"]
        )
        final_mask = self.__get_mask(
            image_h, image_w, anno["mask"]["segmentations"]
        )

        print(visible_mask == 1)
        bbox = mask_to_bbox(visible_mask)
        print(bbox)
        sd_feats = self.__get_feature_from_save(anno["feature_file_name"])
        print(anno["image_height"], anno["image_width"])
        sd_feats = self.__combime_mask_with_sd_features(
            image_height=anno["image_height"],
            image_width=anno["image_width"],
            bbox=bbox,
            sd_features=sd_feats,
        )

        return [visible_mask, invisible_mask, final_mask, bbox, sd_feats]
