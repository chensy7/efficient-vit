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

def init_seed(manual_seed, en_cudnn=False):
    torch.cuda.benchmark = en_cudnn
    torch.cuda.cudnn_enabled = en_cudnn
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    np.random.seed(manual_seed)
    random.seed(manual_seed)


def train(cfg):

    run_id = random.randint(1, 100000)
    init_seed(11733, en_cudnn=True)

    global local_rank
    local_rank = cfg["local_rank"]

    if local_rank == 0:
        logdir = os.path.join("runs", os.path.basename(args.config)[:-4])
        work_dir = os.path.join(logdir, str(run_id))

        if not os.path.exists("runs"):
            os.makedirs("runs")
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)

        shutil.copy(args.config, work_dir)

        logger = get_logger(work_dir)
        logger.info("Let the games begin RUNDIR: {}".format(work_dir))

    # Setup nodes
    torch.cuda.set_device(args.local_rank)

    global gpus_num
    gpus_num = torch.cuda.device_count()
    if local_rank == 0:
        logger.info(f"use {gpus_num} gpus")
        logger.info(f"configure: {cfg}")

    # Setup Augmentations
    train_augmentations = cfg["training"].get("train_augmentations", None)
    t_data_aug = get_composed_augmentations(train_augmentations)
    val_augmentations = cfg["validating"].get("val_augmentations", None)
    v_data_aug = get_composed_augmentations(val_augmentations)

    # Setup Dataloader
    data_loader = get_loader(cfg["data"]["dataset"])
    data_path = cfg["data"]["path"]

    t_loader = data_loader(
        data_path, split=cfg["data"]["train_split"], augmentations=t_data_aug
    )
    v_loader = data_loader(
        data_path, split=cfg["data"]["val_split"], augmentations=v_data_aug
    )
    t_sampler = torch.utils.data.RandomSampler(t_loader)

    trainloader = data.DataLoader(
        t_loader,
        batch_size=cfg["training"]["batch_size"] // gpus_num,
        num_workers=cfg["training"]["n_workers"] // gpus_num,
        shuffle=False,
        sampler=t_sampler,
        pin_memory=True,
        drop_last=True,
    )
    valloader = data.DataLoader(
        v_loader,
        batch_size=cfg["validating"]["batch_size"],
        num_workers=cfg["validating"]["n_workers"],
    )

    if local_rank == 0:
        logger.info("Using training seting {}".format(cfg["training"]))

    # Setup Loss
    loss_fn = get_loss_function(cfg["training"])
    if local_rank == 0:
        logger.info("Using loss {}".format(loss_fn))

    # Setup Model
    model = get_model(cfg["model"], t_loader.n_classes, loss_fn=loss_fn)
    kd_enable = cfg["training"]["teacher"] is not None
    if cfg["training"]["teacher"] is not None:
        print("Knowledge distillation enabled")
        teacher_model = get_model(cfg["training"]["teacher"]["name"], t_loader.n_classes, loss_fn=loss_fn)
        ckpt = torch.load(cfg["training"]["teacher"]["model"])
        teacher_model.load_state_dict(ckpt["model_state"])
        teacher_model.eval()
        teacher_model.cuda()

    # Setup optimizer
    optimizer = get_optimizer(cfg["training"], model)

    # Initialize training param
    start_iter = 0
    best_iou = -100.0

    # Resume from checkpoint
    if cfg["training"]["resume"] is not None and local_rank == 0:
        if os.path.isfile(cfg["training"]["resume"]):
            ckpt = torch.load(cfg["training"]["resume"])
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer"])
            best_iou = ckpt["best_iou"]
            start_iter = ckpt["iter"]
            if local_rank == 0:
                logger.info(
                    "Resuming training from checkpoint '{}'".format(
                        cfg["training"]["resume"]
                    )
                )
        else:
            if local_rank == 0:
                logger.info(
                    "No checkpoint found at '{}'".format(cfg["training"]["resume"])
                )

    model.cuda()

    if local_rank == 0:
        logger.info("Model initialized on GPUs.")
        running_metrics_val = runningScore(t_loader.n_classes)

    time_meter = averageMeter()
    i = start_iter

    writer = SummaryWriter("tensorboard")

    while i <= cfg["training"]["train_iters"]:
        print("{0} out of {1} iterations".format(i, cfg["training"]["train_iters"]))
        for (images, labels) in tqdm(trainloader):
            i += 1
            model.train()
            optimizer.zero_grad()

            start_ts = time.time()
            if kd_enable:
                with torch.no_grad():
                    teacher_outputs = teacher_model(images.cuda())
            else:
                teacher_outputs = None
            loss = model(images.cuda(), labels.cuda(), teacher_outputs)
            loss = torch.mean(loss)
            loss.backward()
            time_meter.update(time.time() - start_ts)

            optimizer.step()

            if local_rank == 0 and (i + 1) % cfg["training"]["print_interval"] == 0:
                fmt_str = "Iter [{:d}/{:d}]  Loss: {:.4f}  Time/Image: {:.4f}"
                print_str = fmt_str.format(
                    i + 1,
                    cfg["training"]["train_iters"],
                    loss.item(),
                    time_meter.avg / cfg["training"]["batch_size"],
                )
                logger.info(print_str)
                time_meter.reset()

            if (
                local_rank == 0
                and (i + 1) % cfg["training"]["val_interval"] == 0
                or (i + 1) == cfg["training"]["train_iters"]
            ):
                model.eval()
                with torch.no_grad():
                    for i_val, (images_val, labels_val) in tqdm(enumerate(valloader)):

                        outputs = model(images_val.cuda())

                        pred = outputs.data.max(1)[1].cpu().numpy()
                        gt = labels_val.data.cpu().numpy()

                        running_metrics_val.update(gt, pred)

                score, class_iou = running_metrics_val.get_scores()
                for k, v in score.items():
                    logger.info("{}: {}".format(k, v))

                for k, v in class_iou.items():
                    logger.info("{}: {}".format(k, v))

                running_metrics_val.reset()

                state = {
                    "iter": i + 1,
                    "model_state": model.state_dict(),
                    "best_iou": score["Mean IoU : \t"],
                    "optimizer": optimizer.state_dict(),
                }
                save_path = os.path.join(
                    work_dir,
                    "{}_{}_last_model.pkl".format(
                        cfg["model"]["arch"], cfg["data"]["dataset"]
                    ),
                )
                torch.save(state, save_path)
                print("Mean IoU {0}, Best {1}".format(score["Mean IoU : \t"], best_iou))

                writer.add_scalar("mean_iou", score["Mean IoU : \t"], i)
                writer.add_scalar("loss", loss, i)

                if score["Mean IoU : \t"] >= best_iou:
                    best_iou = score["Mean IoU : \t"]
                    state = {
                        "iter": i + 1,
                        "model_state": model.state_dict(),
                        "best_iou": best_iou,
                        "optimizer": optimizer.state_dict(),
                    }
                    save_path = os.path.join(
                        work_dir,
                        "{}_{}_best_model.pkl".format(
                            cfg["model"]["arch"], cfg["data"]["dataset"]
                        ),
                    )
                    torch.save(state, save_path)

    writer.close()

if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    class Namespace:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    args = Namespace(config="Uper_resnet18.yml", local_rank=0)

    with open(args.config) as fp:
        cfg = yaml.safe_load(fp)

    cfg["local_rank"] = args.local_rank

    if cfg["training"]["optimizer"]["max_iter"] is not None:
        assert (
            cfg["training"]["train_iters"] == cfg["training"]["optimizer"]["max_iter"]
        )

    train(cfg)
