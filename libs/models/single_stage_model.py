import os
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from libs.models import backbone
from libs import utils


class SingleStageModel(object):

    def __init__(self, params, dist_model=False, rank=None):
        self.model = backbone.__dict__[params["model"]["backbone_arch"]](
            **params["backbone_param"]
        )

        utils.init_weights(self.model, init_type="xavier")

        if dist_model:
            self.model = self.model.to(torch.device(f"cuda:{rank}"))
            self.model = DDP(self.model, device_ids=[rank])
        else:
            self.model.cuda()
            self.model = backbone.FixModule(self.model)
            self.world_size = 1

        if params["optim"] == "SGD":
            self.optim = torch.optim.SGD(
                self.model.parameters(),
                lr=params["lr"],
                momentum=0.9,
                weight_decay=params["weight_decay"],
            )

        elif params["optim"] == "Adam":
            self.optim = torch.optim.Adam(
                self.model.parameters(),
                lr=params["lr"],
                betas=(params["beta1"], 0.999),
            )
        else:
            raise Exception("No such optimizer: {}".format(params["optim"]))

        cudnn.benchmark = True

    def forward_only(self, ret_loss=True):
        pass

    def step(self):
        pass

    def load_state(self, path, Iter=None, resume=False):
        if Iter is not None:
            path = os.path.join(path, "ckpt_iter_{}.pth.tar".format(Iter))

        if resume:
            utils.load_state(path, self.model, self.optim)
        else:
            utils.load_state(path, self.model)

    def load_pretrain(self, load_path):
        utils.load_state(load_path, self.model)

    def save_state(self, path, Iter):
        path = os.path.join(path, "ckpt_iter_{}.pth.tar".format(Iter))

        torch.save(
            {
                "step": Iter,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optim.state_dict(),
            },
            path,
        )

    def switch_to(self, phase):
        if phase == "train":
            self.model.train()
        else:
            self.model.eval()
