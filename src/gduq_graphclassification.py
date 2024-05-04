import os
import numpy as np
import pdb
import tqdm
import time
import torch_geometric as geom

from tap import Tap
from munch import munchify

import torch.nn as nn
import torch

from GOOD.data import load_dataset, create_dataloader
from models.encoders import * 
from GOOD.utils.args import args_parser
from GOOD.utils.config_reader import load_config, args2config, process_configs

from torchmetrics.classification import (
    BinaryCalibrationError,
    MulticlassCalibrationError,
    BinaryAccuracy,
    MulticlassAccuracy,
)

from utils import dUQArgs, DEVICE, count_parameters, test, get_results_summary
from models.gduq_models import GraphANT,GraphANTHiddenReps, GraphANTLayerwiseGIN, GraphANTLayerwisevGIN 
from models.base_models import baseModel, baseModelvGIN, baseModelGINLayerwise, baseModelvGINLayerwise
from models.encoders import GINEncoder, GINEncoderLayerwise, vGINEncoder, vGINEncoderLayerwise 

def main():

    print("=> Begin training")
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

    print("=> dUQ Args: ")
    print("=> Anchor Type: ", config.uq.anchor_type)
    print("=> Num Anchors: ", config.uq.num_anchors)
    print("=> DUQ TYPE: ", config.uq.gduq_type)
    print("=> Save Path: ", config.uq.save_path)
    print("=> Layerwise DUQ: ", config.uq.layerwise_duq)

    """
    Load Data 
    """
    dataset = load_dataset(config.dataset.dataset_name, config)
    if dataset['train'][0].x.shape[1] == 1 and uq_args.gduq_type == 'input':
        print("=> Please use positional encodings an input embedding first.") 
        exit()

    import copy
    # if config.dataset.num_classes == 1:
        #this is a temp patch to avoid a single node output (so I don't have to resize haha)
        #i'll fix this later :) 
        # for k in ['train','val','test','id_val','id_test']:
        #     dataset[k].data.y = copy.deepcopy(torch.nn.functional.one_hot(dataset[k].data.y.long(), num_classes=2)).squeeze()
        # config.dataset.num_classes = 2
    print("=> Num Classes: ",config.dataset.num_classes) 
    print("=> Y Shape: ",dataset['train'][0].y.shape) 
    loader = create_dataloader(dataset, config)
    train_loader = loader["train"]
    id_val = loader["id_val"]
    num_feats = dataset["train"][0].x.shape[1]
    print("=> Num Feats: ", num_feats)
    inputs = next(iter(train_loader))
    inputs = inputs.to(DEVICE)
    print("=> Input Shape: ", inputs.x.shape)
    anchor_type = config.uq.anchor_type
    gduq_type = config.uq.gduq_type

    """
    Create Model.
    """

    if uq_args.gduq_type == 'layerwise':
        if config.model.model_name == "vGIN": 
            print("=> Using vGIN")
            gin_enc = vGINEncoderLayerwise(config)
            base_net = baseModel(encoder=gin_enc, num_classes=config.dataset.num_classes)
        else:
            gin_enc = GINEncoderLayerwise(config)
            print("=> Using GIN")
            base_net = baseModelvGIN(encoder=gin_enc, num_classes=config.dataset.num_classes)
        model = GraphANTLayerwisevGIN(base_network=base_net,
            mean=None,
            std=None,
            anchor_type=anchor_type,
            num_classes=config.dataset.num_classes) 
        
    elif uq_args.gduq_type == 'hidden':
        uq_args.layerwise_duq = -1
        if config.model.model_name == "vGIN": 
            gin_enc = vGINEncoder(config)
        else:
            gin_enc = GINEncoder(config)
        base_net = baseModel(encoder=gin_enc,
            num_classes =config.dataset.num_classes,
            classifier_dim=600)
        model = GraphANTHiddenReps(base_network=base_net,
            mean=None,
            std =None,
            anchor_type=anchor_type,
            num_classes=config.dataset.num_classes)

    elif uq_args.gduq_type == 'input':
        uq_args.layerwise_duq = -1
        x = torch.zeros(len(dataset["train"]), dataset["train"][0].x.shape[1])
        for enum, d in enumerate(dataset["train"]):
            x[enum] = d.x.mean(dim=0)
        x = (x - x.min(dim=0)[0]) / (x.max(dim=0)[0] - x.min(dim=0)[0])
        mu = x.mean(dim=0)
        std = x.std(dim=0)
        if std.norm() < 1e-2:  # if the std is very small, update it.
            std = 0.5
            print("=> Updating Std: ", std)
        config.dataset.dim_node = dataset["train"][0].x.shape[1] * 2  # intentional [x-c,c]
        if config.model.model_name == "vGIN":
            print("=> Using vGIN")
            gin_enc = vGINEncoder(config)
        else:
            gin_enc = GINEncoder(config)
            print("=> Using GIN")

        base_net = baseModel(encoder=gin_enc, num_classes=config.dataset.num_classes)
        model = GraphANT(
            base_network=base_net,
            mean=mu,
            std=std,
            anchor_type=anchor_type,
            num_classes=config.dataset.num_classes,
        )

    else:
        print("=> WARNING WARNING WARNING. INVALID ANCHORING STRATEGY. EXITING!!")
        exit()
    

    model.to(DEVICE)
    model.eval()
    print("=> NUM PARAMETERS: ",count_parameters(model))
    save_name = "_".join(
        [
            "gduq",
            config["dataset"]["dataset_name"],
            config["dataset"]["domain"],
            config["dataset"]["shift_type"],
            config["model"]["model_name"],
            f"{anchor_type}-{uq_args.num_anchors}",
            str(uq_args.layerwise_duq),
            gduq_type, 
            str(config["random_seed"]),
        ]
    )
    print("=> Save Name: ", save_name) 
    """
    Load Data 
    """
    sample = next(iter(train_loader))
    sample = sample.to(DEVICE)
    anchors = model.get_anchors(inputs, num_anchors=config.uq.num_anchors).to(DEVICE)
    print("=> Input Shape: ", inputs.x.shape)
    print("=> Anchors Shape: ", anchors.shape)
    print("=> Save Path: ", uq_args.save_path)

    """
    Set-up Model for Training 
    """
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay
    )
    criterion = torch.nn.CrossEntropyLoss()

    """
    Set-up Metrics
     """
    if config.dataset.num_classes <= 2:
        cal_metric_l1 = BinaryCalibrationError(n_bins=100, norm="l1")
        cal_metric_l2 = BinaryCalibrationError(n_bins=100, norm="l2")  # .to(DEVICE)
        cal_metric_max = BinaryCalibrationError(n_bins=100, norm="max")  # .to(DEVICE)
        acc_metric = BinaryAccuracy(multidim_average='global')
    else:
        cal_metric_l1 = MulticlassCalibrationError(
            num_classes=config.dataset.num_classes, n_bins=100, norm="l1"
        )  
        cal_metric_l2 = MulticlassCalibrationError(
            num_classes=config.dataset.num_classes, n_bins=100, norm="l2"
        )  
        cal_metric_max = MulticlassCalibrationError(
            num_classes=config.dataset.num_classes, n_bins=100, norm="max"
        )  
        acc_metric = MulticlassAccuracy(
            num_classes=config.dataset.num_classes, average="micro"
        )
    
    """
    Training Loop.
    """
    loss_list = []
    acc_list = []
    cal_err_l2_list = []
    train_acc, val_acc = 0, 0
    criterion = config.metric.loss_func
    print("=> CRITERION: ", criterion)
    for epoch in range(config.train.max_epoch):
        loss_avg = 0
        model.train()
        start_time = time.time()
        for batch in train_loader:
            optimizer.zero_grad()
            batch = batch.to(config.device)
            pred = model(batch)
            loss = criterion(pred, batch.y.float())
            loss.backward()
            optimizer.step()
            loss_avg += loss
        loss_avg /= len(train_loader)
        train_acc = test(
            model, loader=loader["eval_train"], acc_metric=acc_metric, anchors=anchors,
            n_anchors=config.uq.num_anchors
        )
        stat_dict = get_results_summary(
            model,
            n_anchors=config.uq.num_anchors,
            anchors=anchors,
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

    anchors = model.get_anchors(inputs, num_anchors=config.uq.num_anchors).to(DEVICE)
    for key in sorted(loader.keys()):
        tmp_loader = loader[key]
        print("====================================")
        if key == 'train':
            pass
        else:
            print("KEY: ", key)
            stat_dict = get_results_summary(
                model,
                n_anchors=config.uq.num_anchors,
                anchors=anchors,
                loader=tmp_loader,
                cal_metric_l1=cal_metric_l1,
                cal_metric_l2=cal_metric_l2,
                cal_metric_max=cal_metric_max,
                acc_metric=acc_metric,
                
            )
            print(f"\tAcc: {stat_dict['acc']:.4f}, CalErrL2: {stat_dict['cal_err_l2']:.4f}")
        print("************************************")
        print()

    """
    Save Model
    """
    ckpt = {
        'mu':model.mean,
        'std':model.std,
        "encoder": model.net.encoder.state_dict(),
        "classifer": model.net.classifier.state_dict(),
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