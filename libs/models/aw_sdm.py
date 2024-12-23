import torch
import torch.nn as nn
import numpy as np

from libs.utils.inference import recover_mask
from . import SingleStageModel


class AWSDM(SingleStageModel):

    def __init__(
        self, params, pretrained_path=None, dist_model=False, rank=None
    ):
        super(AWSDM, self).__init__(
            params=params, dist_model=dist_model, rank=rank
        )
        self.params = params
        self.dist_model = dist_model
        self.use_rgb = params["model"].get("use_rgb", False)

        if pretrained_path is not None:
            self.load_state(pretrained_path)

        self.criterion = nn.CrossEntropyLoss()

    def set_input(self, rgb=None, mask=None, target=None, rank=None):
        device = f"cuda:{rank}" if self.dist_model else "cuda"

        if rgb is not None:
            self.rgb = {}
            for key_i in rgb.keys():
                self.rgb[key_i] = (
                    torch.Tensor(rgb[key_i]).unsqueeze(0).to(device)
                )

        self.mask = torch.Tensor(mask).unsqueeze(0).unsqueeze(0).to(device)
        self.target = torch.Tensor(target).unsqueeze(0).long().to(device)

    def evaluate(self, rgb, mask, bbox, target, rank=None):
        self.set_input(rgb=rgb, mask=mask, target=target, rank=rank)
        output = self.forward_only()
        predict = recover_mask(
            mask=output,
            bbox=bbox,
            height=mask.shape[0],
            width=mask.shape[1],
            interp="linear",
        )

        intersection = ((predict == 1) & (target == 1)).sum()
        predict_area = (predict == 1).sum()
        target_area = (target == 1).sum()
        iou = intersection / (predict_area + target_area - intersection)

        return iou, predict

    def forward_only(self):
        with torch.no_grad():
            if self.use_rgb and self.rgb is not None:
                output = self.model(self.mask, self.rgb)
            else:
                output = self.model(self.mask)

        return output.detach_().argmax(1)[0].cpu().numpy().astype(np.uint8)

    def step(self):
        self.optim.zero_grad()

        if self.use_rgb and self.rgb is not None:
            output = self.model(self.mask, self.rgb)
        else:
            output = self.model(self.mask)

        loss = self.criterion(output, self.target) / self.world_size
        loss.backward()
        self.optim.step()

        return {"loss": loss}
