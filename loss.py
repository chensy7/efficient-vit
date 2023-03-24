import torch
import math
import torch.nn as nn
import logging
import torch
import functools


class CrossEntropyLoss(nn.CrossEntropyLoss):
    """2D Cross Entropy Loss with Auxilary Loss"""

    def __init__(self, weight=None, ignore_index=-1):

        super(CrossEntropyLoss, self).__init__(weight, None, ignore_index)

    def forward(self, pred, target):
        return super(CrossEntropyLoss, self).forward(pred, target)


class OhemCELoss2D(nn.CrossEntropyLoss):
    """2D Cross Entropy Loss with Auxilary Loss"""

    def __init__(self, n_min, thresh=0.7, ignore_index=-1):

        super(OhemCELoss2D, self).__init__(None, None, ignore_index, reduction="none")

        self.thresh = -math.log(thresh)
        self.n_min = n_min
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        return self.OhemCELoss(pred, target)

    def OhemCELoss(self, logits, labels):
        N, C, H, W = logits.size()
        loss = super(OhemCELoss2D, self).forward(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss > self.thresh]
        else:
            loss = loss[: self.n_min]
        return torch.mean(loss)


logger = logging.getLogger("lpcvc")

key2loss = {
    "CrossEntropyLoss": CrossEntropyLoss,
    "OhemCELoss2D": OhemCELoss2D,
}


def get_loss_function(cfg):
    assert cfg["loss"] is not None
    loss_dict = cfg["loss"]
    loss_name = loss_dict["name"]
    loss_params = {k: v for k, v in loss_dict.items() if k != "name"}
    if loss_name not in key2loss:
        raise NotImplementedError("Loss {} not implemented".format(loss_name))

    if loss_name == "OhemCELoss2D":
        n_img_per_gpu = int(cfg["batch_size"] / torch.cuda.device_count())
        cropsize = cfg["train_augmentations"]["rcrop"]
        n_min = n_img_per_gpu * cropsize[0] * cropsize[1] // 16
        loss_params["n_min"] = n_min

    logger.info("Using {} with {} params".format(loss_name, loss_params))
    return key2loss[loss_name](**loss_params)
