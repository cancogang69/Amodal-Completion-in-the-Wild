import torch
import torch.nn as nn
import numpy as np

from libs.utils import average_gradients
from libs.utils.inference import recover_mask
from . import SingleStageModel


class AWSDM(SingleStageModel):

    def __init__(self, params, pretrained_path=None, dist_model=False):
        super(AWSDM, self).__init__(params, dist_model)
        self.params = params
        self.use_rgb = params.get("use_rgb", False)

        if pretrained_path is not None:
            self.load_state(pretrained_path)

        self.criterion = nn.CrossEntropyLoss()

    def set_input(self, rgb=None, mask=None, target=None):
        self.rgb = {}
        for key_i in rgb.keys():
            self.rgb[key_i] = torch.Tensor(rgb[key_i]).unsqueeze(0).cuda()
        self.mask = torch.Tensor(mask).unsqueeze(0).unsqueeze(0).cuda()
        self.target = torch.Tensor(target).unsqueeze(0).long().cuda()

    def evaluate(self, rgb, mask, bbox, target):
        self.set_input(rgb=rgb, mask=mask, target=target)
        output = self.forward_only()
        print(bbox)
        predict = recover_mask(
            mask=output,
            bbox=bbox,
            height=mask.shape[1],
            width=mask.shape[0],
            interp="linear",
        )

        intersection = ((predict == 1) & (target == 1)).sum()
        union = ((predict == 1) | (target == 1)).sum()
        predict_area = (predict == 1).sum()
        target_area = (target == 1).sum()
        iou = intersection / (predict_area + target_area + union)

        return iou

    def forward_only(self):
        with torch.no_grad():
            if self.use_rgb:
                output = self.model(self.mask, self.rgb)
            else:
                output = self.model(self.mask)

        return output.cpu().numpy()

    def step(self):
        if self.use_rgb:
            output = self.model(self.mask, self.rgb)
        else:
            output = self.model(self.mask)
        loss = self.criterion(output, self.target) / self.world_size
        self.optim.zero_grad()
        loss.backward()
        # average_gradients(self.model)
        self.optim.step()
        return {"loss": loss}
