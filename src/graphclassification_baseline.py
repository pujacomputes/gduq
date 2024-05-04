import os
import time
import numpy as np
import pdb
import tqdm

from tap import Tap
from pprint import pprint
from munch import Munch
from munch import munchify

import torch.nn as nn
import torchvision.transforms as transforms
import torch
import torchvision.models as models

from GOOD.data import load_dataset, create_dataloader
from GOOD.networks.models.GINs import GINEncoder
from GOOD.networks.models.GINvirtualnode import vGINEncoder
from GOOD.utils.args import args_parser
from GOOD.utils.config_reader import load_config, args2config, process_configs
from collections import defaultdict

from torchmetrics.classification import (
    BinaryCalibrationError,
    MulticlassCalibrationError,
    BinaryAccuracy,
    MulticlassAccuracy,
    BinaryAUROC
)
from utils import dUQArgs, test, DEVICE, get_results_summary,count_parameters
from models.base_models import baseModel
from models.encoders import GINEncoder, GINEncoderLayerwise, vGINEncoder, vGINEncoderLayerwise 
    
def main():
    """
    Setup Model
    """

    args = args_parser()
    config, duplicate_warnings, duplicate_errors = load_config(args.config_path)
    print("=> Duplicate Warnings: ", duplicate_warnings)
    print("=> Duplicate Errors: ", duplicate_errors)

    args2config(config, args)
    config = munchify(config)
    process_configs(config)

    uq_args = dUQArgs(explicit_bool=True).parse_args(known_only=True)
    config.uq = munchify(uq_args)
    config.model.model_level = "graph"

    """
    Load Data 
    """
    dataset = load_dataset(config.dataset.dataset_name, config)
    loader = create_dataloader(dataset, config)
    train_loader = loader["train"]
    id_val = loader["id_val"]
    num_feats = dataset["train"][0].x.shape[1]
    print("=> Num Feats: ", num_feats)
    inputs = next(iter(train_loader))
    inputs = inputs.to(DEVICE)
    print("=> Input Shape: ", inputs.x.shape)

    """
    Create Model.
    """

    if config.model.model_name == "vGIN":
        print("=> Using vGIN")
        gin_enc = vGINEncoder(config)
    else:
        gin_enc = GINEncoder(config)
        print("=> Using GIN")
    model = baseModel(encoder=gin_enc, num_classes=config.dataset.num_classes)
    model.to(DEVICE)
    model.eval()

    """
    Load Data 
    """
    sample = next(iter(train_loader))
    sample = sample.to(DEVICE)
    print("=> Input Shape: ", inputs.x.shape)
    """
    Set-up Model for Training 
    """
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay
    )

    """
    Set-up Metrics
    """
    if config.dataset.num_classes <= 2:
        cal_metric_l1 = BinaryCalibrationError(n_bins=100, norm="l1")
        cal_metric_l2 = BinaryCalibrationError(n_bins=100, norm="l2")  # .to(DEVICE)
        cal_metric_max = BinaryCalibrationError(n_bins=100, norm="max")  # .to(DEVICE)
        acc_metric = BinaryAccuracy(multidim_average= "global")
        criterion = torch.nn.BCELoss()
    else:
        cal_metric_l1 = MulticlassCalibrationError(
            num_classes=config.dataset.num_classes, n_bins=100, norm="l1"
        )  # .to(DEVICE)
        cal_metric_l2 = MulticlassCalibrationError(
            num_classes=config.dataset.num_classes, n_bins=100, norm="l2"
        )  # .to(DEVICE)
        cal_metric_max = MulticlassCalibrationError(
            num_classes=config.dataset.num_classes, n_bins=100, norm="max"
        )  # .to(DEVICE)
        acc_metric = MulticlassAccuracy(
            num_classes=config.dataset.num_classes, average="micro"
        )
        criterion = torch.nn.CrossEntropyLoss()

    """
    Training Loop.
    """
    loss_list = []
    acc_list = []
    cal_err_l2_list = []
    train_acc, val_acc = 0, 0
    criterion = config.metric.loss_func
    print("=> CRITERION: ", criterion)
    print("=> Num epochs: ", config.train.max_epoch)
    print("=> DEVICE: ",DEVICE)
    print("=> NUM PARAMETERS: ",count_parameters(model))
    for epoch in range(config.train.max_epoch):
        loss_avg = 0
        model.train()
        start_time = time.time()
        for batch in tqdm.tqdm(train_loader,disable=True):
            optimizer.zero_grad()
            batch = batch.to(config.device)
            pred = model(batch)
            loss = criterion(pred, batch.y)
            loss.backward()
            optimizer.step()
            loss_avg += loss
        loss_avg /= len(train_loader)
        train_acc = test(
            model, loader=loader["eval_train"], 
            acc_metric=acc_metric, 
            anchors=None,
            n_anchors=config.uq.num_anchors
        ) 
        stat_dict = get_results_summary(
            model,
            n_anchors=config.uq.num_anchors,
            anchors=None,
            loader=id_val,
            cal_metric_l1=cal_metric_l1,
            cal_metric_l2=cal_metric_l2,
            cal_metric_max=cal_metric_max,
            acc_metric=acc_metric,
        )
        acc_list.append(stat_dict['acc'].cpu())
        cal_err_l2_list.append(stat_dict['cal_err_l2'] .cpu())
        loss_list.append(loss_avg.cpu().detach().numpy())
        end_time = time.time() - start_time 
        print("({times:.3f}) Epoch: {epoch} -- Loss: {loss:.3f} -- Train Acc: {train_acc:.4f} -- Val Acc: {val_acc:.4f} -- CalErrL2: {cal_l2:4f}".format(epoch=epoch,
            loss=loss_avg,
            val_acc=acc_list[-1],
            cal_l2= cal_err_l2_list[-1],
            train_acc=train_acc,
            times= end_time),flush=True)

    for key in sorted(loader.keys()):
        try:
            tmp_loader = loader[key]
            print("====================================")
            print("KEY: ", key)
            stat_dict = get_results_summary(
                model,
                loader=tmp_loader,
                cal_metric_l1=cal_metric_l1,
                cal_metric_l2=cal_metric_l2,
                cal_metric_max=cal_metric_max,
                acc_metric=acc_metric,
            )
            print("acc: ",stat_dict['acc'],"cal_err_l1: ",stat_dict['cal_err_l1'])
            print("************************************")
            print()
        except:
            print("====================================")
            print("SKIPPING: ", key)
            print("************************************")
            print()
    """
    Save Model
    """

    save_name = "_".join(
        [
            "baseline",
            config["dataset"]["dataset_name"],
            config["dataset"]["domain"],
            config["dataset"]["shift_type"],
            config["model"]["model_name"],
            str(config["random_seed"]),
        ]
    )

    ckpt = {
        "encoder": model.encoder.state_dict(),
        "classifer": model.classifier.state_dict(),
        'losses':loss_list,
        'accs':acc_list,
        'cals':cal_err_l2_list
    }
 
    prefix = f"{uq_args.save_path}/{config.dataset.dataset_name}"
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    print("=> Save Name: ", save_name)
    print("=> Save Path: ", prefix)
    torch.save(ckpt, "{}/{}.ckpt".format(prefix, save_name))

if __name__ == "__main__":
    main()