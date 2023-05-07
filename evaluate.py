import os
import oyaml as yaml
import time
import shutil
import torch
import random
import argparse
import numpy as np
import torch.backends.cudnn as cudnn
from torch.utils import data
from tqdm import tqdm

from upernet import get_model
from loss import get_loss_function
from loader import get_loader
from utils import get_logger
from metrics import runningScore, averageMeter
from augmentations import get_composed_augmentations
from optimizers import get_optimizer

from torch.utils.tensorboard import SummaryWriter
from time import time

import prune

def evaluate(cfg):

    # Setup Augmentations
    val_augmentations = cfg["validating"].get("val_augmentations", None)
    v_data_aug = get_composed_augmentations(val_augmentations)

    # Setup Dataloader
    data_loader = get_loader(cfg["data"]["dataset"])
    data_path = cfg["data"]["path"]

    v_loader = data_loader(
        data_path, split=cfg["data"]["val_split"], augmentations=v_data_aug
    )

    valloader = data.DataLoader(
        v_loader,
        batch_size=cfg["validating"]["batch_size"],
        num_workers=cfg["validating"]["n_workers"],
    )

    # Setup Model
    model = get_model(cfg["model"], v_loader.n_classes)
    # prune.prune_model(model.backbone, cfg["model"]["backbone"], 0, True)

    ckpt = torch.load(cfg["validating"]["resume"])
    model.load_state_dict(ckpt["model_state"])

    model = model.cuda().to(torch.float16)
    running_metrics_val = runningScore(v_loader.n_classes)

    model.eval()
    total_time = 0
    with torch.no_grad():
    	for i_val, (images_val, labels_val) in tqdm(enumerate(valloader)):
            images_val = images_val.cuda().to(torch.float16)
            mytime = time()
            outputs = model(images_val)
            total_time += time() - mytime
            pred = outputs.data.max(1)[1].cpu().numpy()
            gt = labels_val.data.cpu().numpy()
            running_metrics_val.update(gt, pred)

    score, class_iou = running_metrics_val.get_scores()
    for k, v in score.items():
    	print("{}: {}".format(k, v))
    for k, v in class_iou.items():
    	print("{}: {}".format(k, v))

    print("FPS: {}".format(len(valloader)/total_time))
    print("Params: {}".format(sum(p.numel() for p in model.parameters())))

if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    class Namespace:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    args = Namespace(config="Uper_resnet18_eval.yml", local_rank=0)

    with open(args.config) as fp:
        cfg = yaml.safe_load(fp)

    evaluate(cfg)